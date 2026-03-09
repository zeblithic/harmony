# aarch64 Boot Stub Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Boot harmony-unikernel on QEMU aarch64 `virt` machine via UEFI, reaching an idle event loop with serial output, heap, identity, and timer.

**Architecture:** New standalone crate `harmony-boot-aarch64` in the harmony-os repo, excluded from workspace. UEFI entry → ExitBootServices → PL011 UART → identity-map MMU → ARM timer → heap → RNDR entropy → Ed25519 identity → idle UnikernelRuntime loop.

**Tech Stack:** Rust no_std, `uefi` crate, `linked_list_allocator`, ARM Generic Timer, PL011 UART, `Aarch64PageTable` from harmony-microkernel.

**Repos:** Design doc + plan live in `zeblithic/harmony`. Code lives in `zeblithic/harmony-os`. A matching branch `jake-boot-aarch64` must exist in both repos.

---

### Task 1: Crate scaffold + UEFI hello world

**Context:** We're creating a new bare-metal crate in harmony-os that boots via UEFI on aarch64. This task just gets the skeleton compiling and printing to the UEFI console. The crate is excluded from the workspace (like harmony-boot for x86_64) because the `aarch64-unknown-uefi` target causes feature unification issues.

**Files:**
- Create: `crates/harmony-boot-aarch64/Cargo.toml`
- Create: `crates/harmony-boot-aarch64/.cargo/config.toml`
- Create: `crates/harmony-boot-aarch64/src/main.rs`

**Step 1: Ensure aarch64-unknown-uefi target is installed**

Run: `rustup target add aarch64-unknown-uefi`
Expected: target installed or already present

**Step 2: Create `.cargo/config.toml`**

```toml
[build]
target = "aarch64-unknown-uefi"

[target.aarch64-unknown-uefi]
rustflags = [
    "--cfg", "poly1305_force_soft",
    "--cfg", "aes_force_soft",
    "--cfg", "curve25519_dalek_backend=\"serial\"",
]
```

**Step 3: Create `Cargo.toml`**

```toml
[package]
name = "harmony-boot-aarch64"
description = "Ring 1: aarch64 boot stub for Harmony unikernel (QEMU virt target)"
version = "0.1.0"
edition = "2021"
license = "GPL-2.0-or-later"
repository = "https://github.com/zeblithic/harmony-os"

# Excluded from workspace — bare-metal target requires independent feature resolution.

[dependencies]
uefi = "0.33"
linked_list_allocator = "0.10"

harmony-unikernel = { path = "../harmony-unikernel", default-features = false }
harmony-identity = { git = "https://github.com/zeblithic/harmony.git", branch = "main", default-features = false }
harmony-platform = { git = "https://github.com/zeblithic/harmony.git", branch = "main", default-features = false }
harmony-microkernel = { path = "../harmony-microkernel", default-features = false }

sha2 = { version = "0.10", default-features = false, features = ["force-soft"] }

[features]
default = []
```

**Important:** Check if `uefi = "0.33"` is the latest stable version on crates.io. If a newer version exists, use it. The API examples below use the 0.33+ API (`uefi::boot`, `uefi::system`). If the API has changed, adapt accordingly — use the context7 MCP tool (`/rust-osdev/uefi-rs`) to look up current API.

**Step 4: Create minimal `src/main.rs`**

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! aarch64 UEFI boot stub for Harmony unikernel.
//!
//! Boots on QEMU `virt` machine. After ExitBootServices, configures PL011
//! UART, identity-maps RAM, enables MMU, and enters an idle event loop.

#![no_main]
#![no_std]

extern crate alloc;

use uefi::prelude::*;
use uefi::{boot, system};
use uefi::mem::memory_map::MemoryType;

#[entry]
fn main() -> Status {
    // Print to UEFI console (works before ExitBootServices)
    system::with_stdout(|stdout| {
        let _ = stdout.output_string(uefi::cstr16!("[UEFI] Booting Harmony aarch64...\r\n"));
    });

    Status::SUCCESS
}
```

**Step 5: Verify it compiles**

Run from `crates/harmony-boot-aarch64/`:
```bash
cargo build
```
Expected: successful build producing `target/aarch64-unknown-uefi/debug/harmony-boot-aarch64.efi`

**Note:** If compilation fails due to dependency resolution (harmony-unikernel or harmony-microkernel pulling in incompatible features), temporarily comment out those dependencies and add them back one at a time to isolate the issue. The `uefi` crate alone should compile cleanly.

**Step 6: Commit**

```bash
git add crates/harmony-boot-aarch64/
git commit -m "feat(boot-aarch64): scaffold UEFI boot crate for QEMU virt"
```

---

### Task 2: PL011 UART driver

**Context:** After ExitBootServices, we lose the UEFI console. PL011 is the UART on QEMU's aarch64 `virt` machine, mapped at physical address `0x0900_0000`. We need TX-only (no RX) for serial output. The driver integrates with harmony-unikernel's `SerialWriter<F>` by providing a `FnMut(u8)` closure.

**Files:**
- Create: `crates/harmony-boot-aarch64/src/pl011.rs`
- Modify: `crates/harmony-boot-aarch64/src/main.rs`

**Step 1: Write the baud rate divisor test**

Add to `src/pl011.rs`:

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! PL011 UART driver — TX-only for serial console output.
//!
//! QEMU `virt` machine maps PL011 at 0x0900_0000 with a 24 MHz reference clock.

/// Compute integer and fractional baud rate divisors.
///
/// Formula: BRD = clock_hz / (16 * baud)
/// IBRD = integer part, FBRD = round(fractional * 64)
pub fn baud_divisors(clock_hz: u32, baud: u32) -> (u16, u8) {
    // Fixed-point: multiply by 4 to get 64ths of the divisor
    // div_x64 = (clock * 4) / baud = (clock / (16 * baud)) * 64
    let div_x64 = (clock_hz as u64 * 4) / baud as u64;
    let ibrd = (div_x64 / 64) as u16;
    let fbrd = (div_x64 % 64) as u8;
    (ibrd, fbrd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baud_115200_at_24mhz() {
        let (ibrd, fbrd) = baud_divisors(24_000_000, 115200);
        // 24_000_000 / (16 * 115200) = 13.0208...
        // IBRD = 13, FBRD = round(0.0208 * 64) = round(1.333) = 1
        assert_eq!(ibrd, 13);
        assert_eq!(fbrd, 1);
    }

    #[test]
    fn baud_9600_at_24mhz() {
        let (ibrd, fbrd) = baud_divisors(24_000_000, 9600);
        // 24_000_000 / (16 * 9600) = 156.25
        // IBRD = 156, FBRD = round(0.25 * 64) = 16
        assert_eq!(ibrd, 156);
        assert_eq!(fbrd, 16);
    }

    #[test]
    fn baud_zero_clock_does_not_panic() {
        let (ibrd, fbrd) = baud_divisors(0, 115200);
        assert_eq!(ibrd, 0);
        assert_eq!(fbrd, 0);
    }
}
```

**Step 2: Run test to verify it compiles and passes**

Run from `crates/harmony-boot-aarch64/`:
```bash
cargo test --target aarch64-apple-darwin -- pl011
```

**Important:** Unit tests run on the HOST, not the UEFI target. You need to run tests with the host target, not `aarch64-unknown-uefi`. Use `cargo test --target aarch64-apple-darwin` (macOS ARM) or just `cargo test` if you temporarily override the default target. If the default target in `.cargo/config.toml` blocks host tests, run: `cargo test --target $(rustc -vV | grep host | cut -d' ' -f2)`

Expected: 3 tests pass

**Step 3: Add the hardware driver code**

Add to `src/pl011.rs` (above the `#[cfg(test)]` block):

```rust
/// QEMU `virt` PL011 base address.
const PL011_BASE: usize = 0x0900_0000;

// Register offsets from PL011 base
const UARTDR: usize = 0x000;    // Data register
const UARTFR: usize = 0x018;    // Flag register
const UARTIBRD: usize = 0x024;  // Integer baud rate divisor
const UARTFBRD: usize = 0x028;  // Fractional baud rate divisor
const UARTLCR_H: usize = 0x02C; // Line control
const UARTCR: usize = 0x030;    // Control register

// Flag register bits
const FR_TXFF: u32 = 1 << 5; // TX FIFO full

// Control register bits
const CR_UARTEN: u32 = 1 << 0; // UART enable
const CR_TXE: u32 = 1 << 8;    // TX enable
const CR_RXE: u32 = 1 << 9;    // RX enable

// Line control bits
const LCR_FEN: u32 = 1 << 4;  // FIFO enable
const LCR_WLEN8: u32 = 3 << 5; // 8-bit word length

/// QEMU `virt` PL011 reference clock (24 MHz).
const PL011_CLOCK_HZ: u32 = 24_000_000;

/// Write a 32-bit value to a PL011 register.
///
/// # Safety
/// Caller must ensure the PL011 MMIO region is mapped and accessible.
unsafe fn write_reg(offset: usize, val: u32) {
    let addr = (PL011_BASE + offset) as *mut u32;
    core::ptr::write_volatile(addr, val);
}

/// Read a 32-bit value from a PL011 register.
///
/// # Safety
/// Caller must ensure the PL011 MMIO region is mapped and accessible.
unsafe fn read_reg(offset: usize) -> u32 {
    let addr = (PL011_BASE + offset) as *const u32;
    core::ptr::read_volatile(addr)
}

/// Initialize PL011 UART: 115200 baud, 8N1, FIFO enabled.
///
/// # Safety
/// Must be called after ExitBootServices and before enabling MMU,
/// OR after MMU is enabled with PL011 region mapped as Device memory.
/// The PL011 base address (0x0900_0000) must be accessible.
pub unsafe fn init() {
    // Disable UART
    write_reg(UARTCR, 0);

    // Set baud rate: 115200 at 24 MHz
    let (ibrd, fbrd) = baud_divisors(PL011_CLOCK_HZ, 115200);
    write_reg(UARTIBRD, ibrd as u32);
    write_reg(UARTFBRD, fbrd as u32);

    // 8-bit word, FIFO enabled
    write_reg(UARTLCR_H, LCR_WLEN8 | LCR_FEN);

    // Enable UART, TX, RX
    write_reg(UARTCR, CR_UARTEN | CR_TXE | CR_RXE);
}

/// Write a single byte to PL011 TX.
///
/// Spins until the TX FIFO has space, then writes the byte.
///
/// # Safety
/// PL011 must be initialized and MMIO region must be accessible.
pub unsafe fn write_byte(byte: u8) {
    // Wait for TX FIFO to have space
    while read_reg(UARTFR) & FR_TXFF != 0 {
        core::hint::spin_loop();
    }
    write_reg(UARTDR, byte as u32);
}
```

**Step 4: Wire PL011 into main.rs**

Update `src/main.rs` — add `mod pl011;` and call init + print after ExitBootServices:

```rust
mod pl011;

use core::fmt::Write;

// ... inside main(), after the UEFI console print and ExitBootServices:

// Exit boot services — we now own the hardware
let memory_map = unsafe { boot::exit_boot_services(MemoryType::LOADER_DATA) };

// Initialize PL011 serial
unsafe { pl011::init() };

// Create a serial writer using PL011
let mut serial = harmony_unikernel::SerialWriter::new(|byte: u8| {
    unsafe { pl011::write_byte(byte) };
});
let _ = writeln!(serial, "[PL011] Serial initialized: 115200 8N1");
```

**Note:** The exact `exit_boot_services` API may differ from the snippet above. Check the uefi crate docs. The key point: call it, get the memory map, then init PL011.

**Step 5: Verify it compiles**

```bash
cargo build
```
Expected: compiles for aarch64-unknown-uefi

**Step 6: Commit**

```bash
git add src/pl011.rs src/main.rs
git commit -m "feat(boot-aarch64): PL011 UART driver for QEMU virt serial output"
```

---

### Task 3: Bump frame allocator

**Context:** The MMU setup (Task 4) needs to allocate 4 KiB frames for intermediate page table levels. We can't use the heap (not initialized yet). A bump allocator is the simplest solution: carve a region from usable memory, hand out frames sequentially, never free. Only used during boot — once the heap is up, we don't allocate more page table frames in this bead's scope.

**Files:**
- Create: `crates/harmony-boot-aarch64/src/bump_alloc.rs`
- Modify: `crates/harmony-boot-aarch64/src/main.rs` (add `mod bump_alloc;`)

**Step 1: Write the tests**

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! Bump frame allocator for boot-time page table construction.
//!
//! Hands out 4 KiB-aligned physical frames from a contiguous region.
//! Never frees — used only during early boot before heap is available.

use harmony_microkernel::vm::{PhysAddr, PAGE_SIZE};

/// A simple bump allocator that hands out 4 KiB frames sequentially.
pub struct BumpAllocator {
    /// Next frame to hand out (always page-aligned).
    next: u64,
    /// One past the last usable byte.
    end: u64,
}

impl BumpAllocator {
    /// Create a new bump allocator over the region `[base, base + size)`.
    ///
    /// `base` must be page-aligned. `size` is rounded down to a page boundary.
    pub fn new(base: u64, size: u64) -> Self {
        let aligned_base = (base + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        let end = base + size;
        // Ensure end doesn't go below aligned_base after rounding
        let end = end & !(PAGE_SIZE - 1);
        Self {
            next: aligned_base,
            end: if end > aligned_base { end } else { aligned_base },
        }
    }

    /// Allocate a single 4 KiB frame, returning its physical address.
    ///
    /// Returns `None` if the allocator is exhausted.
    pub fn alloc_frame(&mut self) -> Option<PhysAddr> {
        if self.next >= self.end {
            return None;
        }
        let frame = self.next;
        self.next += PAGE_SIZE;
        Some(PhysAddr(frame))
    }

    /// Number of frames remaining.
    pub fn remaining(&self) -> u64 {
        (self.end - self.next) / PAGE_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_sequential_frames() {
        let mut alloc = BumpAllocator::new(0x10_0000, 3 * PAGE_SIZE);
        assert_eq!(alloc.remaining(), 3);

        let f1 = alloc.alloc_frame().unwrap();
        assert_eq!(f1, PhysAddr(0x10_0000));

        let f2 = alloc.alloc_frame().unwrap();
        assert_eq!(f2, PhysAddr(0x10_1000));

        let f3 = alloc.alloc_frame().unwrap();
        assert_eq!(f3, PhysAddr(0x10_2000));

        assert_eq!(alloc.remaining(), 0);
        assert!(alloc.alloc_frame().is_none());
    }

    #[test]
    fn unaligned_base_rounds_up() {
        let mut alloc = BumpAllocator::new(0x10_0500, 2 * PAGE_SIZE);
        let f1 = alloc.alloc_frame().unwrap();
        // Base rounds up to 0x10_1000
        assert_eq!(f1, PhysAddr(0x10_1000));
    }

    #[test]
    fn zero_size_returns_none() {
        let mut alloc = BumpAllocator::new(0x10_0000, 0);
        assert!(alloc.alloc_frame().is_none());
        assert_eq!(alloc.remaining(), 0);
    }

    #[test]
    fn size_smaller_than_page_returns_none() {
        let mut alloc = BumpAllocator::new(0x10_0000, 100);
        // 100 bytes < 4096, so after aligning there's no full page
        // Actually base is aligned, but end rounds down: 0x10_0000 + 100 = 0x10_0064
        // end aligned down = 0x10_0000, which == next, so no frames
        assert!(alloc.alloc_frame().is_none());
    }
}
```

**Step 2: Run tests**

```bash
cargo test --target $(rustc -vV | grep host | cut -d' ' -f2) -- bump_alloc
```
Expected: 4 tests pass

**Step 3: Add `mod bump_alloc;` to `main.rs`**

**Step 4: Commit**

```bash
git add src/bump_alloc.rs src/main.rs
git commit -m "feat(boot-aarch64): bump frame allocator for boot-time page tables"
```

---

### Task 4: MMU identity map + enable

**Context:** This is the core challenge. After ExitBootServices, UEFI's page tables are still active but we need our own. We identity-map all usable RAM as Normal cacheable and the PL011 MMIO page as Device-nGnRnE. We use `Aarch64PageTable` from harmony-microkernel (which implements the `PageTable` trait) and our bump allocator for intermediate table frames. After building the map, we configure MAIR_EL1, TCR_EL1, write TTBR0_EL1, and enable MMU via SCTLR_EL1.

**Reference files (read these first):**
- `crates/harmony-microkernel/src/vm/page_table.rs` — `PageTable` trait, especially `map()` signature
- `crates/harmony-microkernel/src/vm/aarch64.rs` — `Aarch64PageTable::new()`, `activate()`, descriptor format
- `crates/harmony-microkernel/src/vm/mod.rs` — `PhysAddr`, `VirtAddr`, `PageFlags`, `PAGE_SIZE`

**Files:**
- Create: `crates/harmony-boot-aarch64/src/mmu.rs`
- Modify: `crates/harmony-boot-aarch64/src/main.rs`

**Key types from harmony-microkernel:**
```rust
// PageTable trait — map() signature:
fn map(
    &mut self,
    vaddr: VirtAddr,
    paddr: PhysAddr,
    flags: PageFlags,
    frame_alloc: &mut dyn FnMut() -> Option<PhysAddr>,
) -> Result<(), VmError>;

// Aarch64PageTable constructor:
unsafe fn new(root: PhysAddr, phys_to_virt: fn(PhysAddr) -> *mut u8) -> Self;

// activate() writes TTBR0_EL1, issues ISB, TLBI, DSB, ISB
fn activate(&mut self);
```

**Step 1: Create `src/mmu.rs` with system register constants and helpers**

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! MMU configuration — identity map + system register setup for aarch64.
//!
//! Builds an identity map (virt == phys) for all usable RAM and device MMIO,
//! then configures MAIR, TCR, and enables the MMU via SCTLR_EL1.

use core::arch::asm;
use core::fmt::Write;

use harmony_microkernel::vm::page_table::PageTable;
use harmony_microkernel::vm::{PageFlags, PhysAddr, VirtAddr, VmError, PAGE_SIZE};

#[cfg(target_arch = "aarch64")]
use harmony_microkernel::vm::aarch64::Aarch64PageTable;

use crate::bump_alloc::BumpAllocator;

/// PL011 UART MMIO base — must be mapped as Device memory.
const PL011_MMIO_BASE: u64 = 0x0900_0000;

// ── MAIR_EL1 ────────────────────────────────────────────────────────

/// Attr0: Normal, Write-Back, Read-Allocate, Write-Allocate (inner + outer).
const MAIR_ATTR0_NORMAL: u64 = 0xFF;
/// Attr1: Device-nGnRnE (no gathering, no reordering, no early ack).
const MAIR_ATTR1_DEVICE: u64 = 0x00;
/// Full MAIR_EL1 value: attr0 in [7:0], attr1 in [15:8].
const MAIR_VALUE: u64 = MAIR_ATTR0_NORMAL | (MAIR_ATTR1_DEVICE << 8);

// ── TCR_EL1 ─────────────────────────────────────────────────────────

/// T0SZ = 16 → 48-bit VA space (64 - 16).
const TCR_T0SZ: u64 = 16;
/// TG0 = 0b00 → 4 KiB granule.
const TCR_TG0_4K: u64 = 0b00 << 14;
/// SH0 = 0b11 → Inner Shareable.
const TCR_SH0_INNER: u64 = 0b11 << 12;
/// ORGN0 = 0b01 → Outer Write-Back, Read-Allocate, Write-Allocate.
const TCR_ORGN0_WB: u64 = 0b01 << 10;
/// IRGN0 = 0b01 → Inner Write-Back, Read-Allocate, Write-Allocate.
const TCR_IRGN0_WB: u64 = 0b01 << 8;
/// Full TCR_EL1 value.
const TCR_VALUE: u64 = TCR_T0SZ | TCR_TG0_4K | TCR_SH0_INNER | TCR_ORGN0_WB | TCR_IRGN0_WB;

// ── SCTLR_EL1 bits ──────────────────────────────────────────────────

const SCTLR_M: u64 = 1 << 0;  // MMU enable
const SCTLR_C: u64 = 1 << 2;  // Data cache enable
const SCTLR_I: u64 = 1 << 12; // Instruction cache enable

/// A memory region descriptor from the UEFI memory map.
#[derive(Clone, Copy)]
pub struct MemoryRegion {
    pub base: u64,
    pub pages: u64,
    pub is_usable: bool,
}

/// Identity-map the given memory regions + PL011 MMIO, configure system
/// registers, and enable the MMU.
///
/// # Safety
///
/// - Must be called at EL1.
/// - `regions` must accurately describe physical memory.
/// - `alloc` must provide valid, zeroed, page-aligned frames.
/// - After this returns, all previously accessible memory remains accessible
///   (identity map), but with proper cache attributes.
#[cfg(target_arch = "aarch64")]
pub unsafe fn init_and_enable(
    regions: &[MemoryRegion],
    alloc: &mut BumpAllocator,
    serial: &mut impl Write,
) {
    // 1. Allocate root page table frame
    let root_frame = alloc
        .alloc_frame()
        .expect("bump allocator exhausted for root frame");

    // Zero the root frame
    let root_ptr = root_frame.as_u64() as *mut u8;
    core::ptr::write_bytes(root_ptr, 0, PAGE_SIZE as usize);

    // 2. Create page table with identity phys_to_virt (phys == virt before MMU switch)
    let mut pt = Aarch64PageTable::new(root_frame, |pa| pa.as_u64() as *mut u8);

    // 3. Create frame allocator closure for map()
    let alloc_ptr = alloc as *mut BumpAllocator;
    let mut frame_alloc = || {
        let frame = (*alloc_ptr).alloc_frame()?;
        // Zero the frame before use as page table
        core::ptr::write_bytes(frame.as_u64() as *mut u8, 0, PAGE_SIZE as usize);
        Some(frame)
    };

    // 4. Map all usable memory regions as Normal cacheable (RW)
    let mut mapped_pages: u64 = 0;
    let flags_rw = PageFlags::READABLE | PageFlags::WRITABLE;

    for region in regions.iter().filter(|r| r.is_usable) {
        for page_idx in 0..region.pages {
            let addr = region.base + page_idx * PAGE_SIZE;
            let result = pt.map(
                VirtAddr(addr),
                PhysAddr(addr),
                flags_rw,
                &mut frame_alloc,
            );
            if let Err(VmError::RegionConflict(_)) = result {
                // Skip overlapping regions (UEFI memory map can have these)
                continue;
            }
            result.expect("failed to map RAM page");
            mapped_pages += 1;
        }
    }

    // 5. Map PL011 MMIO page as Device memory (NO_CACHE flag)
    let flags_device = PageFlags::READABLE | PageFlags::WRITABLE | PageFlags::NO_CACHE;
    pt.map(
        VirtAddr(PL011_MMIO_BASE),
        PhysAddr(PL011_MMIO_BASE),
        flags_device,
        &mut frame_alloc,
    )
    .expect("failed to map PL011 MMIO");

    let _ = writeln!(
        serial,
        "[MMU] Identity mapped {} pages ({} MiB RAM, 1 device page)",
        mapped_pages + 1,
        (mapped_pages * PAGE_SIZE) / (1024 * 1024),
    );

    // 6. Configure system registers
    configure_system_regs(root_frame);

    let _ = writeln!(serial, "[MMU] Enabled: MAIR, TCR, TTBR0, SCTLR");
}

/// Write MAIR_EL1, TCR_EL1, TTBR0_EL1 and enable MMU in SCTLR_EL1.
///
/// # Safety
/// Must be called at EL1 with a valid root page table.
#[cfg(target_arch = "aarch64")]
unsafe fn configure_system_regs(root: PhysAddr) {
    asm!(
        // Drain pending memory operations
        "dsb ish",
        "isb",

        // Set MAIR_EL1
        "msr mair_el1, {mair}",

        // Set TCR_EL1
        "msr tcr_el1, {tcr}",

        // Set TTBR0_EL1 (our page table root)
        "msr ttbr0_el1, {ttbr}",

        // Synchronize
        "isb",

        // Read SCTLR_EL1, enable M + C + I
        "mrs {tmp}, sctlr_el1",
        "orr {tmp}, {tmp}, {sctlr_bits}",
        "msr sctlr_el1, {tmp}",

        // Ensure MMU is active before next instruction
        "isb",

        // Invalidate TLBs
        "tlbi vmalle1is",
        "dsb ish",
        "isb",

        mair = in(reg) MAIR_VALUE,
        tcr = in(reg) TCR_VALUE,
        ttbr = in(reg) root.as_u64(),
        sctlr_bits = in(reg) SCTLR_M | SCTLR_C | SCTLR_I,
        tmp = out(reg) _,
    );
}
```

**Step 2: Wire MMU init into main.rs boot sequence**

After PL011 init and before heap setup, add:

```rust
mod mmu;

use mmu::MemoryRegion;

// ... inside main(), after ExitBootServices and PL011 init:

// Collect UEFI memory map into a fixed-size array
let mut regions = [MemoryRegion { base: 0, pages: 0, is_usable: false }; 128];
let mut region_count = 0;

for desc in memory_map.entries() {
    if region_count >= 128 {
        break;
    }
    let is_usable = matches!(
        desc.ty,
        MemoryType::CONVENTIONAL
            | MemoryType::BOOT_SERVICES_CODE
            | MemoryType::BOOT_SERVICES_DATA
            | MemoryType::LOADER_CODE
            | MemoryType::LOADER_DATA
    );
    regions[region_count] = MemoryRegion {
        base: desc.phys_start,
        pages: desc.page_count,
        is_usable,
    };
    region_count += 1;
}

// Reserve 1 MiB for bump allocator (page table frames)
// Find first usable region >= 1 MiB
let (bump_base, bump_size) = regions[..region_count]
    .iter()
    .filter(|r| r.is_usable && r.pages * 4096 >= 1024 * 1024)
    .map(|r| (r.base, core::cmp::min(r.pages * 4096, 1024 * 1024)))
    .next()
    .expect("no usable memory region >= 1 MiB for bump allocator");

let mut bump = bump_alloc::BumpAllocator::new(bump_base, bump_size);

// Build identity map and enable MMU
unsafe {
    mmu::init_and_enable(&regions[..region_count], &mut bump, &mut serial);
};
```

**Important notes for the implementer:**
- The `Aarch64PageTable` uses `NO_CACHE` flag in `PageFlags` to set AttrIndx=1 (Device memory) in the descriptor. **Read `aarch64.rs` to verify** this mapping. If `NO_CACHE` maps to a different AttrIndx than 1, you may need to adjust the MAIR_EL1 attribute indices.
- UEFI may have left the MMU enabled with its own page tables. Our `configure_system_regs` overwrites TTBR0_EL1 and re-enables — this is safe because both maps are identity maps with the same VA→PA translation.
- The `activate()` method on `Aarch64PageTable` also writes TTBR0_EL1 and does TLB flush. You could use `pt.activate()` INSTEAD of `configure_system_regs()`, but you'd still need to set MAIR_EL1 and TCR_EL1 separately. Decide which approach is cleaner.

**Step 3: Verify it compiles**

```bash
cargo build
```

**Step 4: Commit**

```bash
git add src/mmu.rs src/main.rs
git commit -m "feat(boot-aarch64): MMU identity map + system register configuration"
```

---

### Task 5: ARM generic timer

**Context:** Every ARMv8 core has a built-in counter. No init needed — just read two system registers. We provide `now_ms()` for the event loop's `tick()` calls. Much simpler than x86_64's PIT.

**Files:**
- Create: `crates/harmony-boot-aarch64/src/timer.rs`
- Modify: `crates/harmony-boot-aarch64/src/main.rs` (add `mod timer;`)

**Step 1: Write the conversion tests**

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! ARM Generic Timer — monotonic millisecond clock for the event loop.
//!
//! Reads CNTPCT_EL0 (counter) and CNTFRQ_EL0 (frequency) to provide
//! `now_ms() -> u64`. No initialization required.

/// Convert a counter value to milliseconds given the timer frequency.
///
/// Uses u128 intermediate to avoid overflow. The ARM generic timer
/// frequency is typically 62.5 MHz (QEMU) or 54 MHz (RPi5).
pub fn counter_to_ms(count: u64, freq: u64) -> u64 {
    if freq == 0 {
        return 0;
    }
    ((count as u128 * 1000) / freq as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_count_is_zero_ms() {
        assert_eq!(counter_to_ms(0, 62_500_000), 0);
    }

    #[test]
    fn one_second_at_62_5_mhz() {
        // 62,500,000 ticks at 62.5 MHz = 1000 ms
        assert_eq!(counter_to_ms(62_500_000, 62_500_000), 1000);
    }

    #[test]
    fn large_count_no_overflow() {
        // 24 hours at 62.5 MHz = 5,400,000,000,000 ticks
        let ticks_24h: u64 = 62_500_000 * 3600 * 24;
        let ms = counter_to_ms(ticks_24h, 62_500_000);
        assert_eq!(ms, 24 * 3600 * 1000); // 86,400,000 ms
    }

    #[test]
    fn zero_freq_returns_zero() {
        assert_eq!(counter_to_ms(1000, 0), 0);
    }

    #[test]
    fn fractional_ms_truncates() {
        // 31,250 ticks at 62.5 MHz = 0.5 ms → truncates to 0
        assert_eq!(counter_to_ms(31_250, 62_500_000), 0);
        // 62,500 ticks = 1.0 ms
        assert_eq!(counter_to_ms(62_500, 62_500_000), 1);
    }
}
```

**Step 2: Run tests**

```bash
cargo test --target $(rustc -vV | grep host | cut -d' ' -f2) -- timer
```
Expected: 5 tests pass

**Step 3: Add hardware register reads**

Add to `src/timer.rs` (above the test module):

```rust
use core::arch::asm;

/// Cached timer frequency (read once at boot).
static mut TIMER_FREQ: u64 = 0;

/// Read the timer frequency and cache it.
///
/// # Safety
/// Must be called at EL1 or EL0.
#[cfg(target_arch = "aarch64")]
pub unsafe fn init() {
    let freq: u64;
    asm!("mrs {}, cntfrq_el0", out(reg) freq);
    TIMER_FREQ = freq;
}

/// Return the cached timer frequency in Hz.
pub fn freq() -> u64 {
    unsafe { TIMER_FREQ }
}

/// Read the current counter value.
///
/// # Safety
/// Timer must be initialized.
#[cfg(target_arch = "aarch64")]
pub fn counter() -> u64 {
    let count: u64;
    unsafe { asm!("mrs {}, cntpct_el0", out(reg) count) };
    count
}

/// Current monotonic time in milliseconds since boot.
///
/// # Safety
/// `init()` must have been called first.
#[cfg(target_arch = "aarch64")]
pub fn now_ms() -> u64 {
    counter_to_ms(counter(), freq())
}
```

**Step 4: Wire into main.rs**

```rust
mod timer;

// ... inside main(), after MMU enable:

unsafe { timer::init() };
let _ = writeln!(serial, "[Timer] ARM generic timer: freq={} Hz", timer::freq());
```

**Step 5: Commit**

```bash
git add src/timer.rs src/main.rs
git commit -m "feat(boot-aarch64): ARM generic timer for monotonic milliseconds"
```

---

### Task 6: RNDR entropy

**Context:** ARMv8.5 adds the RNDR register — a hardware true random number generator. QEMU supports it with `-cpu max`. We need to fill byte buffers for `KernelEntropy`, which takes a `fn(&mut [u8])` closure for Ed25519/X25519 keypair generation.

**Files:**
- Create: `crates/harmony-boot-aarch64/src/rndr.rs`
- Modify: `crates/harmony-boot-aarch64/src/main.rs` (add `mod rndr;`)

**Step 1: Write the buffer fill test**

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! RNDR hardware random number generator (ARMv8.5-RNG).
//!
//! QEMU supports RNDR with `-cpu max`. Each read of the RNDR system
//! register returns 64 bits of hardware entropy.

/// Fill a byte buffer from 64-bit random values.
///
/// `read_u64` is called repeatedly to get 8 bytes at a time. Any
/// trailing bytes (if `buf.len()` is not a multiple of 8) are filled
/// from the last 64-bit read.
pub fn fill_from_u64(buf: &mut [u8], mut read_u64: impl FnMut() -> u64) {
    let mut offset = 0;
    while offset + 8 <= buf.len() {
        let val = read_u64();
        buf[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
        offset += 8;
    }
    if offset < buf.len() {
        let val = read_u64();
        let remaining = &val.to_le_bytes()[..buf.len() - offset];
        buf[offset..].copy_from_slice(remaining);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_exact_multiple_of_8() {
        let mut buf = [0u8; 16];
        let mut counter = 0u64;
        fill_from_u64(&mut buf, || {
            counter += 1;
            counter
        });
        // First 8 bytes = 1u64 LE, next 8 bytes = 2u64 LE
        assert_eq!(&buf[0..8], &1u64.to_le_bytes());
        assert_eq!(&buf[8..16], &2u64.to_le_bytes());
    }

    #[test]
    fn fill_non_multiple_of_8() {
        let mut buf = [0u8; 5];
        fill_from_u64(&mut buf, || 0x0807060504030201u64);
        assert_eq!(buf, [0x01, 0x02, 0x03, 0x04, 0x05]);
    }

    #[test]
    fn fill_empty_buffer() {
        let mut buf = [0u8; 0];
        fill_from_u64(&mut buf, || panic!("should not be called"));
    }

    #[test]
    fn fill_single_byte() {
        let mut buf = [0u8; 1];
        fill_from_u64(&mut buf, || 0xABu64);
        assert_eq!(buf, [0xAB]);
    }
}
```

**Step 2: Run tests**

```bash
cargo test --target $(rustc -vV | grep host | cut -d' ' -f2) -- rndr
```
Expected: 4 tests pass

**Step 3: Add hardware RNDR read + detection**

Add to `src/rndr.rs` (above the test module):

```rust
use core::arch::asm;

/// Check if RNDR is available by reading ID_AA64ISAR0_EL1 bits [63:60].
///
/// # Safety
/// Must be called at EL1 (EL0 may not have access to this register).
#[cfg(target_arch = "aarch64")]
pub unsafe fn is_available() -> bool {
    let isar0: u64;
    asm!("mrs {}, id_aa64isar0_el1", out(reg) isar0);
    // RNDR field is bits [63:60], value >= 1 means RNDR is supported
    (isar0 >> 60) & 0xF >= 1
}

/// Read a single 64-bit random value from the RNDR register.
///
/// Retries on failure (NZCV.Z set means retry needed).
///
/// # Safety
/// RNDR must be available (check with `is_available()` first).
#[cfg(target_arch = "aarch64")]
pub unsafe fn read_u64() -> u64 {
    loop {
        let val: u64;
        let success: u64;
        asm!(
            "mrs {val}, s3_3_c2_c4_0",  // RNDR
            "cset {success}, ne",         // NE = success
            val = out(reg) val,
            success = out(reg) success,
        );
        if success != 0 {
            return val;
        }
        // Yield before retry
        asm!("yield");
    }
}

/// Fill a byte buffer with hardware random data from RNDR.
///
/// # Safety
/// RNDR must be available.
#[cfg(target_arch = "aarch64")]
pub unsafe fn fill(buf: &mut [u8]) {
    fill_from_u64(buf, || read_u64());
}
```

**Step 4: Wire into main.rs**

```rust
mod rndr;

// ... inside main(), after timer init:

// Verify RNDR hardware RNG is available
assert!(unsafe { rndr::is_available() }, "RNDR not available — use QEMU with -cpu max");
let _ = writeln!(serial, "[RNDR] Hardware RNG available");
```

**Step 5: Commit**

```bash
git add src/rndr.rs src/main.rs
git commit -m "feat(boot-aarch64): RNDR hardware entropy for identity generation"
```

---

### Task 7: Heap + identity + idle event loop

**Context:** Final integration. Initialize the heap allocator, generate an Ed25519/X25519 identity, create a `UnikernelRuntime`, and enter an idle event loop. This proves the entire sans-I/O stack works on aarch64.

**Reference files (read these first):**
- `crates/harmony-boot/src/main.rs:228-280` — how x86_64 boot does heap + identity + runtime
- `crates/harmony-unikernel/src/event_loop.rs` — `UnikernelRuntime` API
- `crates/harmony-unikernel/src/platform/entropy.rs` — `KernelEntropy` constructor
- `crates/harmony-unikernel/src/platform/persistence.rs` — `MemoryState` constructor

**Files:**
- Modify: `crates/harmony-boot-aarch64/src/main.rs`

**Step 1: Set up global allocator**

At the top of `main.rs`, before `#[entry]`:

```rust
use linked_list_allocator::LockedHeap;

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();
```

**Step 2: Add heap initialization after MMU enable**

```rust
// ... after MMU enable, timer init, RNDR check:

// Find largest usable memory region for heap (skip the bump allocator region)
let (heap_base, heap_size) = regions[..region_count]
    .iter()
    .filter(|r| r.is_usable && r.base != bump_base)
    .map(|r| (r.base, r.pages * PAGE_SIZE))
    .max_by_key(|(_, size)| *size)
    .expect("no usable memory region for heap");

// Cap heap at 4 MiB
let heap_size = core::cmp::min(heap_size, 4 * 1024 * 1024);

unsafe {
    ALLOCATOR.lock().init(heap_base as *mut u8, heap_size as usize);
}
let _ = writeln!(serial, "[Heap] Initialized: {} bytes at {:#x}", heap_size, heap_base);
```

**Step 3: Generate identity**

```rust
use harmony_identity::Identity;
use harmony_unikernel::{KernelEntropy, MemoryState, UnikernelRuntime};

// ... after heap init:

// Create entropy source using RNDR
let entropy = KernelEntropy::new(|buf: &mut [u8]| {
    unsafe { rndr::fill(buf) };
});

// Generate Ed25519/X25519 keypair
let identity = Identity::generate(&entropy);
let _ = writeln!(
    serial,
    "[Identity] Generated Ed25519 address: {:02x}{:02x}{:02x}{:02x}...",
    identity.address()[0],
    identity.address()[1],
    identity.address()[2],
    identity.address()[3],
);
```

**Important:** The `Identity` and `KernelEntropy` APIs may differ from this snippet. Read the actual source files to get the correct constructors and method names. The x86_64 boot's `main.rs` lines 264-280 show the working pattern — mirror it.

**Step 4: Create runtime and enter idle loop**

```rust
// Create persistent state store
let persistence = MemoryState::new();

// Create unikernel runtime (no network interfaces registered)
let mut runtime = UnikernelRuntime::new(identity, entropy, persistence);
let _ = writeln!(serial, "[Runtime] UnikernelRuntime created, entering idle loop");

// Idle event loop — tick the runtime, handle actions, sleep
loop {
    let now = timer::now_ms();

    // Tick the runtime
    let actions = runtime.tick(now);
    for action in actions {
        // No network interfaces to act on — just log if needed
        let _ = writeln!(serial, "[Runtime] action: {:?}", action);
    }

    // Wait for event (ARM equivalent of HLT — saves power vs spinning)
    unsafe { core::arch::asm!("wfe") };
}
```

**Important:** The `UnikernelRuntime::new()`, `tick()` API, and return types may differ. Read `crates/harmony-unikernel/src/event_loop.rs` for the actual interface. The x86_64 boot's main loop (lines 595-616) shows the working pattern.

**Step 5: Verify it compiles**

```bash
cargo build
```

**Step 6: Add a panic handler if not already present**

The `uefi` crate may provide one. If not, add to `main.rs`:

```rust
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    // Try to print panic info over serial
    let mut serial = harmony_unikernel::SerialWriter::new(|byte: u8| {
        unsafe { pl011::write_byte(byte) };
    });
    let _ = writeln!(serial, "\n!!! PANIC: {}", info);
    loop {
        unsafe { core::arch::asm!("wfe") };
    }
}
```

**Step 7: Commit**

```bash
git add src/main.rs
git commit -m "feat(boot-aarch64): heap + identity + idle event loop — full boot sequence"
```

---

## QEMU Verification (After All Tasks)

This is the acceptance test — not automated, run manually.

**Build the EFI binary:**
```bash
cd crates/harmony-boot-aarch64
cargo build
```

**Set up ESP directory structure:**
```bash
mkdir -p /tmp/harmony-esp/EFI/BOOT
cp target/aarch64-unknown-uefi/debug/harmony-boot-aarch64.efi \
   /tmp/harmony-esp/EFI/BOOT/BOOTAA64.EFI
```

**Get AAVMF firmware (one-time):**
```bash
# macOS (Homebrew)
brew install qemu   # includes AAVMF at /opt/homebrew/share/qemu/edk2-aarch64-code.fd

# Or download directly from https://retrage.github.io/edk2-nightly/
```

**Run QEMU:**
```bash
qemu-system-aarch64 \
  -machine virt \
  -cpu max \
  -m 128M \
  -nographic \
  -serial stdio \
  -bios /opt/homebrew/share/qemu/edk2-aarch64-code.fd \
  -drive format=raw,file=fat:rw:/tmp/harmony-esp
```

**Expected output:**
```
[UEFI] Booting Harmony aarch64...
[PL011] Serial initialized: 115200 8N1
[MMU] Identity mapped N pages (X MiB RAM, 1 device page)
[MMU] Enabled: MAIR, TCR, TTBR0, SCTLR
[Timer] ARM generic timer: freq=62500000 Hz
[RNDR] Hardware RNG available
[Heap] Initialized: 4194304 bytes at 0x...
[Identity] Generated Ed25519 address: abcd...
[Runtime] UnikernelRuntime created, entering idle loop
```

**Kill QEMU:** Ctrl+A then X.

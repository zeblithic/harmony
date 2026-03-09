# aarch64 Boot Stub Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Boot harmony-unikernel on QEMU aarch64 `virt` machine via UEFI, reaching an idle event loop with serial output, heap, identity, and timer.

**Architecture:** New standalone crate `harmony-boot-aarch64` targeting `aarch64-unknown-uefi`. UEFI firmware provides initial environment; after `ExitBootServices()` we configure PL011 UART, identity-map RAM via our existing `Aarch64PageTable`, enable MMU, init heap and timer, generate an Ed25519 identity, and enter an idle `UnikernelRuntime` loop.

**Tech Stack:** Rust (`no_std`), `uefi` crate, ARM Generic Timer, PL011 UART, `Aarch64PageTable` from harmony-microkernel, `linked_list_allocator`.

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dev target | QEMU `virt` first | Fast iteration, no hardware needed. RPi5 drivers are a separate bead. |
| Crate structure | Separate `harmony-boot-aarch64` | No shared code with x86_64 boot — shared logic already lives in harmony-unikernel. |
| Boot protocol | UEFI via `uefi` crate | RPi5 uses UEFI firmware, so no throwaway work. Mature Rust ecosystem. |
| Scope | Boot → serial → heap → identity → idle loop | Proves full sans-I/O stack on aarch64. No networking (separate bead). |

## Crate Structure

```
crates/harmony-boot-aarch64/
├── Cargo.toml
├── .cargo/
│   └── config.toml          # target = aarch64-unknown-uefi
├── src/
│   ├── main.rs              # UEFI entry, boot sequence orchestration
│   ├── pl011.rs             # PL011 UART driver (QEMU virt @ 0x0900_0000)
│   ├── mmu.rs               # System register setup (MAIR, TCR, SCTLR), identity map
│   ├── bump_alloc.rs        # Boot-time bump frame allocator for page tables
│   ├── timer.rs             # ARM generic timer (CNTP_* registers)
│   └── rndr.rs              # ARMv8.5 RNDR hardware entropy
```

**Not in workspace** — excluded like x86_64 `harmony-boot` to avoid feature unification with the bare-metal target.

### Dependencies

- `uefi = "0.33"` — UEFI entry point, memory map, console
- `linked_list_allocator = "0.10"` — heap allocator (same as x86_64)
- `harmony-unikernel` — `UnikernelRuntime`, `KernelEntropy`, `MemoryState`, `SerialWriter`
- `harmony-identity` — keypair generation
- `harmony-platform` — `NetworkInterface` trait
- `harmony-microkernel` — `Aarch64PageTable`, `PageTable` trait, `PageFlags`
- `sha2` with `force-soft` — cross-compile safety

## Boot Sequence

Ten steps after UEFI entry:

1. **UEFI console** — Print "Booting Harmony aarch64..." via UEFI SimpleTextOutput. Useful if PL011 setup fails.
2. **UEFI memory map** — Collect conventional memory regions (base + size) before losing access.
3. **ExitBootServices()** — We now own the hardware. No more UEFI calls.
4. **PL011 UART init** — Configure PL011 at `0x0900_0000`: 115200 baud, 8N1, FIFO enabled. All subsequent output goes here.
5. **Identity-map RAM + MMIO** — Use `Aarch64PageTable` to map UEFI memory regions as Normal cacheable and PL011 page as Device-nGnRnE. Bump allocator for page table frames.
6. **Enable MMU** — Configure MAIR_EL1, TCR_EL1, TTBR0_EL1, then set SCTLR_EL1.M/C/I.
7. **ARM generic timer** — Read CNTFRQ_EL0 for frequency. Provide `now_ms()` via CNTPCT_EL0.
8. **Heap init** — Largest usable RAM region, cap at 4 MiB, `LockedHeap::init()`.
9. **Identity generation** — RNDR register (ARMv8.5) for entropy. Detect via ID_AA64ISAR0_EL1, panic if unavailable. Generate Ed25519/X25519 keypair.
10. **Idle event loop** — `UnikernelRuntime::new()`, loop: `tick(now_ms)` → dispatch actions → `WFE`.

## PL011 UART Driver

Minimal TX-only driver for QEMU `virt` PL011 at `0x0900_0000`.

| Offset | Register | Purpose |
|--------|----------|---------|
| 0x000 | UARTDR | Write byte to transmit |
| 0x018 | UARTFR | Flags — bit 5 (TXFF) = TX FIFO full |
| 0x024 | UARTIBRD | Integer baud divisor |
| 0x028 | UARTFBRD | Fractional baud divisor |
| 0x02C | UARTLCR_H | Line control (word length, FIFO) |
| 0x030 | UARTCR | Control (enable, TX/RX enable) |

**Init:** Disable UART → set baud (IBRD=13, FBRD=1 for 115200 at 24 MHz) → 8-bit FIFO (LCR_H=0x70) → enable (CR=0x301).

**Write:** Spin on UARTFR.TXFF, write byte to UARTDR.

**Integration:** Closure `|byte: u8|` for `SerialWriter<F>` from harmony-unikernel.

## MMU Setup

### MAIR_EL1

- Attr0 (index 0) = `0xFF` — Normal, Write-Back cacheable (inner + outer). For RAM.
- Attr1 (index 1) = `0x00` — Device-nGnRnE. For MMIO.

### TCR_EL1

- T0SZ = 16 → 48-bit VA space
- TG0 = 0b00 → 4 KiB granule
- SH0 = 0b11 → Inner Shareable
- ORGN0 = 0b01 → Outer WB cacheable
- IRGN0 = 0b01 → Inner WB cacheable
- EPD1 = 1 → Disable TTBR1 walks (upper VA half unused)
- IPS = runtime from ID_AA64MMFR0_EL1.PARange (3-bit mask)

### Identity Map Strategy

- Walk UEFI memory map: map conventional memory as Normal (AttrIndx=0)
- Map PL011 page (0x0900_0000, 4 KiB) as Device (AttrIndx=1)
- Use `Aarch64PageTable` from harmony-microkernel for page table manipulation
- Bump allocator for intermediate page table frames (carved from usable memory, pre-heap)

### Enable Sequence

```
DSB ISH + ISB
TLBI vmalle1is + DSB ISH + ISB   (invalidate stale UEFI TLB entries first)
Read ID_AA64MMFR0_EL1.PARange → set TCR_EL1.IPS
Write MAIR_EL1
Write TCR_EL1 (with runtime IPS)
Write TTBR0_EL1
ISB
Read/modify/write SCTLR_EL1: set M, C, I
ISB
```

## Timer & Entropy

### ARM Generic Timer

No init required — built into every ARMv8 core.

- `CNTFRQ_EL0` — frequency in Hz (read once at boot)
- `CNTPCT_EL0` — current count (read each loop iteration)
- `now_ms = (CNTPCT_EL0 as u128 * 1000 / CNTFRQ_EL0 as u128) as u64`

### RNDR Entropy

ARMv8.5 hardware RNG. QEMU supports with `-cpu max`.

- Read RNDR via `mrs` — 64 bits per read
- Check NZCV for success, retry on failure
- Detect support via `ID_AA64ISAR0_EL1` bits [63:60], panic if absent
- Provides `fn(&mut [u8])` closure for `KernelEntropy`

## Testing

### Unit Tests (~5-8)

- Baud rate divisor calculation (pure math)
- Timer millisecond conversion (overflow edge cases, u128 intermediate)
- RNDR buffer fill logic (mock register reads)

### QEMU Acceptance Test (Manual)

```bash
qemu-system-aarch64 \
  -machine virt \
  -cpu max \
  -m 128M \
  -nographic \
  -serial stdio \
  -bios /path/to/AAVMF_CODE.fd \
  -drive format=raw,file=fat:rw:esp/
```

Where `esp/EFI/BOOT/BOOTAA64.EFI` is the built binary.

**Expected output:**
```
[UEFI] Booting Harmony aarch64...
[PL011] Serial initialized: 115200 8N1
[MMU] Identity mapped N pages (X MiB RAM, 1 device page)
[MMU] Enabled: MAIR, TCR, TTBR0, SCTLR
[Timer] ARM generic timer: freq=62500000 Hz
[RNDR] Hardware RNG available
[Heap] Initialized: 4194304 bytes at 0x...
[Identity] Generated Ed25519 address: <hex>
[Runtime] UnikernelRuntime created, entering idle loop
```

## Out of Scope

- Networking / VirtIO MMIO driver → `harmony-haq`
- RPi5 hardware drivers → `harmony-b9e`
- Ring 2/3 integration → separate beads
- CI automation for QEMU boot test → future infrastructure work
- TTBR1 (kernel upper half) configuration → Ring 2/3 scope

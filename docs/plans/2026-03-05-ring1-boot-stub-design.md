# Ring 1: Minimal x86_64 Boot Stub (Multiboot2) — Design

**Date:** 2026-03-05
**Bead:** harmony-cpo
**Status:** Approved
**Branch:** jake-os-boot-stub

## Problem

The Harmony unikernel (Ring 1) needs a minimal bootable x86_64 image that proves the Ring 0 sans-I/O state machines work on bare metal. This is the "hello world" of Ring 1 — boot to a serial console with identity announcement, no networking.

## Approach

**Controlled boot**: Use the `bootloader` crate for the multiboot2/page-table/GDT ceremony. Everything after the Rust entry point is ours: serial, heap, IDT, platform trait impls, event loop.

## Crate Structure

```
harmony-os/
├── Cargo.toml                    (workspace — adds harmony-boot)
├── justfile                      (build + QEMU recipes)
├── crates/
│   ├── harmony-boot/             (NEW — x86_64 binary crate)
│   │   ├── Cargo.toml
│   │   ├── .cargo/config.toml    (default target + runner)
│   │   └── src/
│   │       └── main.rs           (#![no_std] #![no_main] entry point)
│   ├── harmony-unikernel/        (lib — portable kernel logic)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── event_loop.rs     (UnikernelRuntime, tick loop)
│   │       ├── platform/
│   │       │   ├── mod.rs
│   │       │   ├── entropy.rs    (KernelEntropy<F> — EntropySource impl)
│   │       │   └── persistence.rs (MemoryState — PersistentState impl)
│   │       └── serial.rs         (SerialWriter<F> — structured output)
│   ├── harmony-microkernel/      (unchanged)
│   └── harmony-os/               (unchanged)
```

**Dependency flow:**
```
harmony-boot (binary, x86_64-specific)
  ├── harmony-unikernel (portable kernel logic)
  │   ├── harmony-crypto      (no_std, from core)
  │   ├── harmony-identity    (no_std, from core)
  │   ├── harmony-platform    (no_std, traits)
  │   └── harmony-reticulum   (no_std, from core)
  ├── bootloader_api           (BootInfo struct, entry_point! macro)
  ├── linked_list_allocator    (global heap allocator)
  └── x86_64                   (IDT, port I/O for UART, RDRAND)
```

`harmony-boot` owns x86_64-specific things (UART ports, RDRAND, IDT, heap init from boot memory map). `harmony-unikernel` is arch-agnostic — it takes trait impls and runs the event loop.

## Boot Sequence

```
BIOS/UEFI -> bootloader crate (page tables, GDT, long mode)
    |
    v
kernel_main(boot_info: &'static mut BootInfo)
    |
    +-- 1. Init serial (write to I/O port 0x3F8)
    |      -> [BOOT] Harmony unikernel v0.1.0
    |
    +-- 2. Init heap allocator
    |      -> find usable memory region from boot_info.memory_regions
    |      -> init linked_list_allocator with fixed-size region (1MB)
    |      -> [HEAP] initialized 1048576 bytes
    |
    +-- 3. Init IDT
    |      -> double fault, breakpoint, page fault handlers
    |      -> [IDT] loaded
    |
    +-- 4. Init RDRAND entropy source
    |      -> verify RDRAND support via CPUID
    |      -> wrap in EntropySource impl
    |      -> [ENTROPY] RDRAND available
    |
    +-- 5. Generate identity
    |      -> PrivateIdentity::generate(&mut entropy_source)
    |      -> derive address hash
    |      -> [IDENTITY] <32-char hex address>
    |
    +-- 6. Build platform + enter event loop
    |      -> create in-memory PersistentState
    |      -> construct Node with identity (no network interfaces)
    |      -> [READY] entering event loop
    |      -> loop { node.tick(); hlt(); }
    |
    +-- 7. (unreachable) [HALT] shutting down
```

Panic handler prints `[PANIC] <message>` to serial, then loops `hlt` forever.

## Platform Trait Implementations

All in `harmony-unikernel`, arch-agnostic via dependency injection.

### EntropySource — `KernelEntropy<F>`

Wraps a caller-provided `FnMut(&mut [u8])` closure. `harmony-boot` passes RDRAND; a future aarch64 crate passes `MRS RNDR`. Also implements `rand_core::RngCore + CryptoRng` so it works with `PrivateIdentity::generate()`.

### PersistentState — `MemoryState`

In-memory `BTreeMap<String, Vec<u8>>`. No actual persistence — data lost on reboot. Identity regenerated each boot. Uses `BTreeMap` (from `alloc`) to avoid extra dependencies.

Feature flag support for compile-time baked identity to be added later.

### Serial — `SerialWriter<F>`

Arch-agnostic writer taking a `FnMut(u8)` closure. `harmony-boot` passes UART `0x3F8` port write. Implements `core::fmt::Write`. Provides `log(tag, msg)` helper for structured `[TAG] message` output.

### What stays in harmony-boot (x86_64-specific)

| Component | Implementation |
|-----------|---------------|
| UART init | Configure 0x3F8 line control, baud rate, FIFO |
| RDRAND | `x86_64::instructions::random::RdRand` -> closure for KernelEntropy |
| IDT | `x86_64::structures::idt::InterruptDescriptorTable` with exception handlers |
| Heap | `linked_list_allocator::LockedHeap` as `#[global_allocator]`, init from memory map |

## Event Loop

```rust
pub struct UnikernelRuntime<E: EntropySource, P: PersistentState> {
    node: harmony_reticulum::Node,
    identity: harmony_identity::PrivateIdentity,
    entropy: E,
    persistence: P,
    tick_count: u64,
}
```

`tick(&mut self) -> bool`: polls events (none yet), feeds into Node, executes actions (no-op without interfaces), returns whether work was done.

Even with no network interfaces, the event loop proves:
- Ring 0 state machines initialize in bare-metal context
- `alloc`-dependent types work with our heap
- Platform trait impls compile and wire up
- Sans-I/O pattern works end-to-end

## Testing

### Unit Tests (host-native)

`cargo test -p harmony-unikernel` — no QEMU needed:

| Test | Verifies |
|------|----------|
| `KernelEntropy` with mock fill_fn | EntropySource produces random bytes |
| `MemoryState` CRUD | PersistentState round-trips correctly |
| `SerialWriter` with captured bytes | Structured `[TAG] message` format |
| `UnikernelRuntime::new()` + `tick()` | Runtime initializes, tick returns false |

### QEMU Integration Test

Justfile recipe boots image, checks serial output:

```
just test-qemu
```

- Boots with `-serial stdio -display none`
- `timeout 10` kills on hang
- `isa-debug-exit` device for clean QEMU shutdown
- Greps for `[IDENTITY]` line — exit 0 = pass

### Serial Output Contract

```
[BOOT] Harmony unikernel v0.1.0
[HEAP] initialized 1048576 bytes
[IDT] loaded
[ENTROPY] RDRAND available
[IDENTITY] a1b2c3d4e5f6...  (32 hex chars = 128-bit address)
[READY] entering event loop
```

## Scope Boundaries

### Delivers

1. `harmony-boot` crate — x86_64 bootable binary
2. `harmony-unikernel` crate — expanded with portable kernel logic
3. Build tooling — justfile, .cargo/config.toml
4. Tests — unit tests + QEMU integration test

### Does NOT deliver

| Out of scope | Deferred to |
|-------------|-------------|
| VirtIO-net driver | harmony-bpo |
| Actual packet routing | After VirtIO-net |
| Persistent identity (flash/block) | Future bead |
| aarch64 / SUNDRAGON support | Future harmony-boot-aarch64 |
| Async executor | Future — current loop is synchronous tick() + hlt() |
| smoltcp IP stack | After VirtIO-net |

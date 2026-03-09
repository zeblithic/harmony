# Aarch64 Syscall Trap Design

## Summary

Add aarch64 SVC-based syscall trapping to the Harmony OS Linuxulator, mirroring
the existing x86_64 syscall path. Introduces a CPU-agnostic `LinuxSyscall` enum
for cross-architecture dispatch, an exception vector table with diagnostic abort
handlers, and EM_AARCH64 support in the ELF loader.

**Bead:** harmony-kax

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Execution model | EL1 flat | Everything at EL1, SVC from EL1->EL1. Matches x86_64 Ring 0 flat model. MVP simplicity. |
| Vector table scope | SVC + abort handlers | 16 entries total; only 0x200 (Current EL SPx Sync) has dispatch. Abort handlers print FAR/ESR diagnostics. Other 15 entries panic. Near-zero marginal cost for significant debugging value. |
| Syscall mapping | `LinuxSyscall` enum | Per-arch mapping functions (`from_aarch64`, `from_x86_64`) translate raw numbers to shared enum. Linuxulator matches on enum variants, not magic numbers. Clean for future arches (RISC-V). |
| ELF loader | Extend in this bead | Add `EM_AARCH64` (0xB7) acceptance via `cfg(target_arch)`. Rest of ET_EXEC loader is already arch-agnostic. Gives end-to-end testable path. |
| Test strategy | Host unit tests only | Pure logic (dispatch table, enum mapping, trap frame layout) tested on host. Hardware-dependent code is `cfg(target_arch = "aarch64")`, verified by manual QEMU boot. QEMU CI is a follow-up. |

## Module Layout

```
harmony-os/src/
  linuxulator.rs    — LinuxSyscall enum, from_aarch64(), from_x86_64(),
                      dispatch refactored to match on enum variants
  elf.rs            — add EM_AARCH64, cfg-gated machine check

harmony-boot-aarch64/src/
  vectors.rs  (new) — VBAR_EL1 vector table (asm), init_vectors()
  syscall.rs  (new) — TrapFrame, svc_handler(), abort_handler()
  main.rs           — wire up vectors + Linuxulator + ELF load
```

## Component Details

### 1. LinuxSyscall Enum (harmony-os/src/linuxulator.rs)

CPU-agnostic syscall representation. Each architecture maps native numbers into
this enum before the Linuxulator dispatches.

```rust
pub enum LinuxSyscall {
    Read { fd: i32, buf: u64, count: u64 },
    Write { fd: i32, buf: u64, count: u64 },
    Open { pathname: u64, flags: i32, mode: u32 },
    Close { fd: i32 },
    Brk { addr: u64 },
    Mmap { addr: u64, len: u64, prot: i32, flags: i32, fd: i32, offset: u64 },
    Munmap { addr: u64, len: u64 },
    ExitGroup { code: i32 },
    ArchPrctl { code: i32, addr: u64 },
    SetTidAddress { tidptr: u64 },
    // ... remaining syscalls the Linuxulator handles
    Unknown { nr: u64 },
}

pub fn from_x86_64(nr: u64, args: [u64; 6]) -> LinuxSyscall { ... }
pub fn from_aarch64(nr: u64, args: [u64; 6]) -> LinuxSyscall { ... }
```

The existing `dispatch_syscall` match on raw x86_64 numbers becomes a match on
`LinuxSyscall` variants. Identical `SyscallBackend` calls underneath.

### 2. Exception Vector Table (harmony-boot-aarch64/src/vectors.rs)

2048-byte aligned table at VBAR_EL1. 16 entries, 128 bytes each.

| Offset | Context | Type | Handler |
|---|---|---|---|
| 0x200 | Current EL, SPx | Synchronous | Real dispatch (SVC/abort) |
| All others | — | — | `unexpected_exception` panic |

The 0x200 entry:
1. Saves X0-X30 + ELR_EL1 + SPSR_EL1 to stack (TrapFrame)
2. Reads ESR_EL1 for exception class (EC)
3. EC=0x15 (SVC AArch64) -> `svc_handler`
4. EC=0x25 (Data Abort) / EC=0x21 (Instruction Abort) -> `abort_handler`
5. Other EC -> panic

Installation:
```rust
pub unsafe fn init() {
    core::arch::asm!("msr vbar_el1, {}", in(reg) &VECTOR_TABLE as *const _ as u64);
    core::arch::asm!("isb");
}
```

### 3. SVC Handler (harmony-boot-aarch64/src/syscall.rs)

```rust
#[repr(C)]
pub struct TrapFrame {
    pub x: [u64; 31],   // X0-X30
    pub elr: u64,        // saved PC
    pub spsr: u64,       // saved PSTATE
}

pub extern "C" fn svc_handler(frame: &mut TrapFrame) {
    let nr = frame.x[8];                                     // X8 = syscall nr
    let args = [frame.x[0], frame.x[1], frame.x[2],
                frame.x[3], frame.x[4], frame.x[5]];        // X0-X5
    let syscall = LinuxSyscall::from_aarch64(nr, args);
    let result = LINUXULATOR.dispatch(syscall);               // i64
    frame.x[0] = result as u64;                              // return in X0
}
```

Register convention:

| | x86_64 | aarch64 |
|---|---|---|
| Syscall nr | RAX | X8 |
| Args | RDI,RSI,RDX,R10,R8,R9 | X0-X5 |
| Return | RAX | X0 |
| Instruction | `syscall` | `SVC #0` |

### 4. Abort Handlers (harmony-boot-aarch64/src/syscall.rs)

Diagnostic handlers for Data Abort (EC=0x25) and Instruction Abort (EC=0x21).
Print FAR_EL1 (faulting address), ELR_EL1 (faulting PC), and ESR_EL1 (syndrome)
to PL011 serial via `panic!()`, then halt.

### 5. ELF Loader (harmony-os/src/elf.rs)

Add `EM_AARCH64 = 0xB7`. Machine check becomes:

```rust
let expected_machine = if cfg!(target_arch = "x86_64") {
    EM_X86_64
} else if cfg!(target_arch = "aarch64") {
    EM_AARCH64
} else {
    return Err(ElfError::UnsupportedMachine);
};
```

Rest of ET_EXEC loader unchanged — segment loading and entry point are
arch-agnostic.

### 6. Boot Integration (harmony-boot-aarch64/src/main.rs)

Three new steps after existing Runtime creation:

1. `vectors::init()` — install exception vector table
2. Initialize Linuxulator with SyscallBackend, store in static
3. Optionally load embedded test ELF binary (`include_bytes!`), jump to entry

Idle WFE loop remains as fallback after binary exits.

## Test Plan

All tests run on host via `cargo test` — no QEMU required.

| Test | Crate | Verifies |
|---|---|---|
| `from_aarch64_write` | harmony-os | nr=64 -> Write variant |
| `from_aarch64_read` | harmony-os | nr=63 -> Read variant |
| `from_aarch64_exit_group` | harmony-os | nr=94 -> ExitGroup variant |
| `from_aarch64_unknown` | harmony-os | nr=9999 -> Unknown variant |
| `from_x86_64_write` | harmony-os | nr=1 -> Write (regression) |
| `dispatch_write_calls_backend` | harmony-os | Write variant calls backend |
| `elf_accepts_aarch64` | harmony-os | EM_AARCH64 accepted |
| `elf_rejects_wrong_machine` | harmony-os | Wrong arch rejected |
| `trap_frame_size` | boot-aarch64 | sizeof matches asm (264 bytes) |
| `trap_frame_x8_offset` | boot-aarch64 | X8 at byte 64 |
| `trap_frame_elr_offset` | boot-aarch64 | ELR at correct offset |

~11 host-runnable tests total. QEMU integration testing deferred to future bead.

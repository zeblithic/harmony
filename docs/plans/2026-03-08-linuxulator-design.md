# Linuxulator MVP — ELF Loader + Syscall Translation for Ring 3

**Date:** 2026-03-08
**Status:** Approved
**Scope:** harmony-os (crates/harmony-os, crates/harmony-boot)

## Problem

Ring 3 of harmony-os is an empty scaffold. The first Ring 3 capability is the Linuxulator — a syscall translation layer that lets Linux binaries run on the Harmony microkernel by translating Linux syscalls into 9P operations against existing FileServers.

The MVP target: a statically-linked x86_64 assembly binary that calls `write(1, "Hello\n", 6)` and `exit_group(0)`, with output routed through the 9P SerialServer to the QEMU serial console.

## Architecture

### Overview

```
Linux binary: syscall instruction
       ↓
x86_64 MSR trap (harmony-boot, naked asm)
       ↓
Linuxulator::handle_syscall(nr=1, args=[1, ptr, 6])
       ↓
match nr { 1 => sys_write, 231 => sys_exit_group, _ => -ENOSYS }
       ↓
sys_write: fd_table[1] → fid → backend.write(fid, 0, data)
       ↓
Kernel dispatches 9P write to SerialServer
       ↓
Bytes appear on QEMU serial console
```

Three layers with clean separation:
1. **ELF loader** (`harmony-os::elf`) — pure parsing, no hardware deps
2. **Linuxulator** (`harmony-os::linuxulator`) — syscall→9P translation, fd table
3. **Syscall trap** (`harmony-boot::syscall`) — x86_64 MSR setup and naked asm handler

## ELF Loader

### Scope

Statically-linked x86_64 ELF executables only (ET_EXEC). No dynamic linking, no PT_INTERP, no shared libraries. Rejects anything else with a clear error.

### Parsing

Reads the ELF64 header for:
- `e_entry` — virtual address of entry point
- Program headers — PT_LOAD segments to map

For each PT_LOAD segment:
- `p_vaddr` — requested virtual address
- `p_filesz` — bytes to copy from ELF file
- `p_memsz` — total memory size (difference is zero-filled, for .bss)
- `p_offset` — offset into the ELF file

### Memory Model (MVP)

No MMU, no page tables. Flat address space (everything runs in Ring 0). The loader:

1. Validates ELF magic, class (64-bit), machine (x86_64), type (ET_EXEC)
2. Iterates PT_LOAD segments
3. For each: allocates `p_memsz` bytes from heap, copies `p_filesz` bytes, zero-fills remainder
4. Records mapping: `p_vaddr → heap_address`
5. Allocates stack (64 KiB)
6. Returns `ElfImage` with entry point, segments, and stack pointer

The test binary uses RIP-relative addressing, so it works regardless of where segments are loaded. Fixed-address binaries require MMU support (future work).

### API

```rust
pub struct ElfImage {
    pub entry_point: u64,
    pub segments: Vec<LoadedSegment>,
    pub stack_top: u64,
}

pub struct LoadedSegment {
    pub vaddr: u64,
    pub loaded_at: *mut u8,
    pub memsz: usize,
    pub flags: SegmentFlags,
}

pub fn load_elf(elf_bytes: &[u8]) -> Result<ElfImage, ElfError>;
```

## Linuxulator

### SyscallBackend Trait

```rust
pub trait SyscallBackend {
    fn walk(&mut self, path: &str, new_fid: Fid) -> Result<QPath, IpcError>;
    fn open(&mut self, fid: Fid, mode: OpenMode) -> Result<(), IpcError>;
    fn read(&mut self, fid: Fid, offset: u64, count: u32) -> Result<Vec<u8>, IpcError>;
    fn write(&mut self, fid: Fid, offset: u64, data: &[u8]) -> Result<u32, IpcError>;
    fn clunk(&mut self, fid: Fid) -> Result<(), IpcError>;
}
```

Mirrors the Kernel's IPC methods without needing a PID. Unit tests provide a mock. Production wraps `Kernel` calls with the Linux process's PID.

### Linuxulator Struct

```rust
pub struct Linuxulator<B: SyscallBackend> {
    backend: B,
    fd_table: BTreeMap<i32, Fid>,
    next_fid: Fid,
    exit_code: Option<i32>,
}
```

### Initialization

Before jumping to the ELF entry point, the Linuxulator pre-populates the fd table:
- **fd 0** (stdin) → fid walked to `/dev/serial`, opened Read (reads return empty for MVP)
- **fd 1** (stdout) → fid walked to `/dev/serial`, opened Write
- **fd 2** (stderr) → same fid as fd 1

Requires SerialServer mounted at `/dev/serial` in the process's namespace.

### Syscall Dispatch

```rust
pub fn handle_syscall(&mut self, nr: u64, args: [u64; 6]) -> i64 {
    match nr {
        1   => self.sys_write(args[0] as i32, args[1], args[2]),
        231 => self.sys_exit_group(args[0] as i32),
        _   => -38, // -ENOSYS
    }
}
```

### sys_write

Looks up fd in fd_table, reads bytes from process memory (flat address space — raw pointer dereference), calls `backend.write(fid, 0, data)`. Returns bytes written or negative errno.

### sys_exit_group

Sets `self.exit_code = Some(code)`. Caller checks this flag after `handle_syscall` returns and halts execution.

### MVP Syscalls

| Nr | Name | Behavior |
|----|------|----------|
| 1 | write | fd→fid lookup, 9P write to backend |
| 231 | exit_group | Set exit flag, halt |
| * | unknown | Return -ENOSYS |

## x86_64 Syscall Trap

Lives in `harmony-boot::syscall`. Architecture-specific — the only part that can't be unit-tested on the host.

### MSR Setup

Before jumping to the ELF entry point:
- **IA32_EFER** (0xC0000080) — Set bit 0 (SCE, System Call Enable)
- **IA32_LSTAR** (0xC0000082) — Address of `syscall_entry` handler
- **IA32_STAR** (0xC0000081) — Kernel CS/SS in bits 47:32
- **IA32_FMASK** (0xC0000084) — Clear IF on syscall entry

### Syscall Entry (naked asm)

```
syscall_entry:
    ; RCX = return RIP, R11 = return RFLAGS (saved by CPU)
    ; RAX = syscall nr, RDI/RSI/RDX/R10/R8/R9 = args 1-6

    Save callee-saved registers + RCX + R11
    Shuffle registers to build args array
    Call dispatch_syscall(nr, &args) -> i64
    Move result to RAX
    Restore registers
    sysretq
```

### dispatch_syscall

```rust
extern "C" fn dispatch_syscall(nr: u64, args: &[u64; 6]) -> i64 {
    // Access global Linuxulator (see below)
    let result = LINUXULATOR.handle_syscall(nr, *args);
    if LINUXULATOR.exited() {
        // Log exit code, halt or return to kernel_main
    }
    result
}
```

### Global State

The syscall handler needs access to the Linuxulator instance. Since the `syscall` instruction provides no mechanism to pass context, we use a `static mut` behind an unsafe accessor. Safe because: single CPU, interrupts disabled during handler, one Linux process.

## Execution Flow

```
kernel_main():
    ... existing boot (serial, heap, entropy, identity, virtio) ...

    #[cfg(feature = "ring3")] {
        // 1. Set up Ring 2 kernel
        let kernel = Kernel::new(identity);

        // 2. Spawn SerialServer, mount at /dev/serial
        let serial_pid = kernel.spawn_process("serial", Box::new(SerialServer::new(...)));
        let linux_pid = kernel.spawn_process("hello", ..., &[("/dev/serial", serial_pid, 0)]);
        kernel.grant_endpoint_cap(&mut entropy, linux_pid, serial_pid, now);

        // 3. Create Linuxulator with kernel backend
        let backend = KernelBackend::new(linux_pid, &mut kernel);
        let mut linuxulator = Linuxulator::new(backend);
        linuxulator.init_stdio();  // pre-open fd 0/1/2

        // 4. Load ELF
        let elf_bytes = include_bytes!("../../test-bins/hello.elf");
        let image = load_elf(elf_bytes).expect("ELF load failed");

        // 5. Set up MSRs, install global linuxulator
        setup_syscall_msrs();
        set_global_linuxulator(linuxulator);

        // 6. Jump to binary
        jump_to_elf(image.entry_point, image.stack_top);

        // 7. Returns here after exit_group
        serial.log("LINUX", &format!("exit_group({})", get_exit_code()));
    }
```

## Module Layout

### harmony-os crate

```
crates/harmony-os/src/
├── lib.rs              (pub mod elf; pub mod linuxulator;)
├── elf.rs              (ELF parser: load_elf, ElfImage, ElfError)
└── linuxulator.rs      (Linuxulator, SyscallBackend, fd table, handle_syscall)
```

Both modules are `no_std` compatible. Fully testable via `cargo test -p harmony-os`.

### harmony-boot crate

```
crates/harmony-boot/
├── src/
│   ├── main.rs          (add ring3 block gated on feature)
│   └── syscall.rs       (MSR setup, naked asm, dispatch_syscall, global state)
└── test-bins/
    ├── hello.S          (source)
    └── hello.elf        (pre-built binary)
```

### Dependencies

```
harmony-boot [ring3 feature]
  ├── harmony-os           ← ELF loader, Linuxulator
  ├── harmony-microkernel  ← Kernel, FileServer, SerialServer
  └── harmony-unikernel    ← event loop, serial, platform
```

`ring3` implies `ring2` (Linuxulator needs the Kernel for 9P dispatch).

## Test Binary

```asm
; hello.S — minimal Linux x86_64 binary
.globl _start
_start:
    mov rax, 1          ; syscall: write
    mov rdi, 1          ; fd: stdout
    lea rsi, [rip+msg]  ; buf: pointer to message
    mov rdx, 6          ; count: 6 bytes
    syscall

    mov rax, 231        ; syscall: exit_group
    xor rdi, rdi        ; code: 0
    syscall

msg: .ascii "Hello\n"
```

Cross-compiled: `as -o hello.o hello.S && ld -o hello.elf -nostdlib --static hello.o`

## Testing

### Layer 1: ELF parser unit tests (`cargo test -p harmony-os`)
- Parse valid ELF, verify header fields, segment count, entry point
- Reject bad magic, 32-bit, non-x86_64, dynamic (ET_DYN)
- Edge cases: zero segments, oversized segment

### Layer 2: Linuxulator unit tests (`cargo test -p harmony-os`)
- Mock SyscallBackend that records calls
- `handle_syscall(1, [1, ptr, 6, ...])` → write called with correct fid/data
- `handle_syscall(231, [0, ...])` → exit flag set
- Unknown syscall → -ENOSYS
- Bad fd → -EBADF
- fd table init: fd 0/1/2 mapped correctly

### Layer 3: QEMU integration test
- Boot with `ring3` feature
- Serial output contains "Hello"
- Clean exit via QEMU debug exit

## Success Criteria

QEMU serial output shows:
```
[BOOT] Harmony unikernel v0.1.0
...
[KERN] Ring 3 Linuxulator mode
[LINUX] loading hello.elf (N bytes, 1 segment)
[LINUX] entry=0x... stack=0x...
Hello
[LINUX] exit_group(0)
```

## Future Work (NOT in scope)

- Hardware process isolation (page tables, Ring 3 ↔ Ring 0 separation)
- Dynamic linking, PT_INTERP, shared libraries
- Additional syscalls (open, close, mmap, brk, arch_prctl, etc.)
- C runtime support (musl libc)
- aarch64 port
- ContentServer-backed filesystem
- Multiple concurrent Linux processes

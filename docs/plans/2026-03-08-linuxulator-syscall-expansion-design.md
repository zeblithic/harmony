# Linuxulator Syscall Expansion — musl libc Hello World

**Date:** 2026-03-08
**Status:** Approved
**Scope:** harmony-os (crates/harmony-os, crates/harmony-boot)
**Bead:** harmony-ujq

## Problem

The Linuxulator MVP (harmony-0h0) supports 2 syscalls: `write` and `exit_group`. A hand-written assembly binary can print "Hello\n", but no C runtime can boot. musl libc's `_start` → `__libc_start_main` → `main` path requires ~15-18 syscalls before reaching `main()`: TLS setup (`arch_prctl`), heap probing (`brk`), memory allocation (`mmap`), and various runtime queries.

**Goal:** A statically-linked musl `hello.c` (`printf("Hello\n"); return 0;`) runs to completion on the Harmony microkernel, with output on the QEMU serial console.

## Architecture

### Approach: Monolithic Expansion

All syscall logic stays in `linuxulator.rs`. A new `MemoryArena` struct is added as a field on `Linuxulator`. The `handle_syscall` match grows from 2 to ~18 arms. File grows from ~450 to ~900 lines.

Split into modules deferred until complexity warrants it (YAGNI).

### Memory Model: 4KB-Page Arena

Linux binaries do raw pointer arithmetic (`mov [rax+8], rbx`). Memory must be directly addressable — no indirection layer. But the arena's internal structure aligns with Harmony's CAS chunk size (4KB) so the bridge to content-addressed storage is natural when VM support arrives (harmony-qv2).

```
Arena layout (1 MiB default):

  base                                              base + size
  |                                                           |
  [  brk region →→→→→→→→→  |  free  |  ←←←← mmap region    ]
  |                         |                                 |
  brk_offset                              mmap_top
```

- **brk** grows upward from offset 0. `sys_brk(0)` returns base address (probe). `sys_brk(addr)` extends, 4KB-aligned.
- **mmap(MAP_ANONYMOUS)** grows downward from arena ceiling. Returns 4KB-aligned addresses.
- **munmap** is a no-op (bump allocator). Real reclamation deferred to VM layer.
- brk and mmap cannot collide — each checks the other's boundary.

### CAS Alignment (Future Bridge)

The 4KB page granularity is deliberate. When VM support (harmony-qv2) adds page tables:
- Each 4KB arena page maps 1:1 to a CAS chunk
- Page fault → fetch chunk from ContentServer → map into address space
- Process suspend → hash each dirty page → store as chunk → get 32-bit address
- Sub-page packing (2KB, 1KB, ..., 32B) per athenaeum logic for final snapshots

This layer is the point where "pure CAS zen" breaks down — mutable, directly-addressed memory while the process runs. The alignment preserves the bridge for later.

## MemoryArena

```rust
pub struct MemoryArena {
    pages: Vec<u8>,        // contiguous backing memory
    base: usize,           // pages.as_ptr() as usize
    brk_offset: usize,     // current brk (offset from base, grows up)
    mmap_regions: Vec<(usize, usize)>,  // (offset, length) tracking
    mmap_top: usize,       // next mmap boundary (grows down from size)
}
```

Pre-allocates the full arena as a single `Vec<u8>`. The Linux process sees `base` through `base + size` as its address space. All allocations are 4KB-aligned.

## Syscall Table

### Fully Implemented

| Nr | Name | Behavior |
|----|------|----------|
| 0 | read | fd→fid lookup, backend.read, copy into process memory |
| 1 | write | (existing) fd→fid lookup, backend.write |
| 3 | close | fd→fid lookup, backend.clunk, remove from fd table |
| 5 | fstat | fd→fid lookup, backend.stat, write Linux `struct stat` (144 bytes) |
| 9 | mmap | MAP_ANONYMOUS only — carve from arena top, zero-fill, 4KB-aligned |
| 11 | munmap | no-op (accept and return 0, don't reclaim) |
| 12 | brk | probe (addr=0) or extend heap, 4KB-aligned, returns new brk |
| 60 | exit | same as exit_group (single process) |
| 158 | arch_prctl | ARCH_SET_FS: wrfsbase/wrmsr for TLS. ARCH_GET_FS: read back. |
| 231 | exit_group | (existing) set exit flag |
| 257 | openat | read path from process memory, walk+open via backend, allocate fd |

### Stubs (return success, no real work)

| Nr | Name | Return | Why |
|----|------|--------|-----|
| 13 | rt_sigaction | 0 | no signals in single-process model |
| 14 | rt_sigprocmask | 0 | no signals |
| 16 | ioctl | -ENOTTY for TIOCGWINSZ, -ENOSYS otherwise | musl probes terminal |
| 218 | set_tid_address | 1 (fake TID) | musl startup, no threads |
| 273 | set_robust_list | 0 | futex cleanup, not needed |
| 302 | prlimit64 | 0 + write rlimit for RLIMIT_STACK | musl queries stack size |
| 334 | rseq | -ENOSYS | musl handles gracefully |

## IpcError → errno Mapping

Centralized helper so every sys_* method uses consistent error translation:

```
IpcError::NotFound        → -2  (ENOENT)
IpcError::PermissionDenied → -13 (EACCES)
IpcError::NotOpen         → -9  (EBADF)
IpcError::InvalidFid      → -9  (EBADF)
IpcError::NotDirectory    → -20 (ENOTDIR)
IpcError::IsDirectory     → -21 (EISDIR)
IpcError::ReadOnly        → -30 (EROFS)
IpcError::ResourceExhausted → -12 (ENOMEM)
IpcError::Conflict        → -17 (EEXIST)
IpcError::NotSupported    → -38 (ENOSYS)
IpcError::InvalidArgument → -22 (EINVAL)
```

## arch_prctl and FS Base

musl calls `arch_prctl(ARCH_SET_FS, tls_addr)` to set the Thread Local Storage base. On x86_64, this writes the FS segment base register. Implementation in `harmony-boot::syscall`:

- Use `wrfsbase` instruction if CPUID reports FSGSBASE support
- Fallback to `wrmsr(IA32_FS_BASE = 0xC0000100, addr)`

The Linuxulator calls into a platform-specific function exposed by harmony-boot. This is the one new piece of architecture-specific code beyond the existing MSR setup.

## Linux struct stat Layout

`fstat` must write a 144-byte `struct stat` at the user-provided pointer. Key fields for musl:

```
offset  size  field       value for stdio fds
0       8     st_dev      0
8       8     st_ino      qpath from FileStat
16      8     st_nlink    1
24      4     st_mode     S_IFCHR | 0o666 (for stdio), S_IFREG | 0o644 (regular files)
28      4     st_uid      0
32      4     st_gid      0
36      4     (pad)       0
40      8     st_rdev     0
48      8     st_size     FileStat.size
56      8     st_blksize  4096
64      8     st_blocks   (size + 511) / 512
72-144        timestamps  0 (all zeros for MVP)
```

Stdio fds (0, 1, 2) report `S_IFCHR` (character device). Files opened via openat report `S_IFREG` or `S_IFDIR` based on `FileStat.file_type`.

## Testing

### Layer 1: MemoryArena unit tests
- brk(0) returns base
- brk(base + N) extends, returns new brk, 4KB-aligned
- mmap allocates from top, returns valid address
- brk can't exceed mmap_top (returns current brk)
- mmap can't exceed brk_offset (returns -ENOMEM)
- Arena exhaustion

### Layer 2: Syscall unit tests (MockBackend)
- sys_read: copies data from backend into process memory
- sys_close: removes fd, clunks fid
- sys_openat: walks path, opens, returns new fd
- sys_fstat: writes correct struct stat layout
- sys_brk: probe and extend
- sys_mmap: anonymous allocation, zero-filled
- sys_arch_prctl: SET_FS/GET_FS round-trip (host-only if FSGSBASE available)
- All stubs: return expected values
- ipc_err_to_errno: all variants mapped correctly

### Layer 3: Integration test (KernelBackend)
- Expand existing test: open file via openat → write → read back → fstat → close
- Full fd lifecycle through real Kernel + SerialServer

### Layer 4: QEMU smoke test
- Cross-compile `printf("Hello\n"); return 0;` with `musl-gcc -static`
- Embed in harmony-boot, boot in QEMU
- Verify "Hello" on serial output
- Manual test (not in cargo test), documented in test-bins/README

## Module Layout

No new files. Expanded existing files:

```
crates/harmony-os/src/
├── lib.rs              (unchanged)
├── elf.rs              (unchanged)
└── linuxulator.rs      (expanded: MemoryArena + ~16 new sys_* methods + tests)

crates/harmony-boot/src/
├── main.rs             (add arena init, update dispatch for arch_prctl)
└── syscall.rs          (add write_fs_base/read_fs_base)
```

## Success Criteria

1. `cargo test -p harmony-os` — all new syscall unit tests pass
2. `cargo test -p harmony-os` — arena allocation tests pass
3. `cargo clippy --workspace` — zero warnings
4. QEMU boot with musl-static hello prints "Hello" on serial console

## Future Work (NOT in scope)

- File-backed mmap (needs ContentServer integration)
- Multiple concurrent processes / threads
- Real signal delivery
- Virtual memory / page tables (harmony-qv2)
- Dynamic linking / shared libraries (harmony-ms6)
- aarch64 syscall trap (harmony-kax)

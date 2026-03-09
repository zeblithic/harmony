# Aarch64 Syscall Trap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add aarch64 SVC-based syscall trapping to the Linuxulator, with a cross-architecture `LinuxSyscall` enum, exception vector table, and EM_AARCH64 ELF support.

**Architecture:** The Linuxulator gains a CPU-agnostic `LinuxSyscall` enum. Each architecture provides a mapping function (`from_x86_64`, `from_aarch64`) that converts raw syscall numbers to enum variants. The aarch64 boot crate gets a VBAR_EL1 exception vector table with SVC dispatch and diagnostic abort handlers. The ELF loader accepts EM_AARCH64.

**Tech Stack:** Rust (no_std), aarch64 inline assembly, QEMU virt platform

**Repos:** Changes span two repos:
- `harmony-os` (`/Users/zeblith/work/zeblithic/harmony-os`) — harmony-os crate (LinuxSyscall enum, ELF) + harmony-boot-aarch64 crate (vectors, syscall handler)
- `harmony` (`/Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-kax-syscall-trap`) — this design doc + plan

**Branch:** `jake-kax-syscall-trap` (already exists in both repos)

**Test command:** harmony-os workspace tests: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test --workspace`
harmony-boot-aarch64 tests (excluded from workspace): `cd /Users/zeblith/work/zeblithic/harmony-os/crates/harmony-boot-aarch64 && cargo test --target $(rustc -vV | grep host | cut -d' ' -f2)`

---

### Task 1: LinuxSyscall enum and from_x86_64 mapping

Add the `LinuxSyscall` enum to `linuxulator.rs` and a `from_x86_64` mapping function that preserves the current x86_64 syscall number semantics. This task does NOT change the dispatch yet — it just adds the new types alongside the existing code.

**Files:**
- Modify: `harmony-os/crates/harmony-os/src/linuxulator.rs`

**Step 1: Write the failing tests**

Add these tests at the bottom of the existing `#[cfg(test)] mod tests` block in `linuxulator.rs`:

```rust
#[test]
fn from_x86_64_write() {
    let syscall = LinuxSyscall::from_x86_64(1, [1, 0x1000, 5, 0, 0, 0]);
    match syscall {
        LinuxSyscall::Write { fd, buf, count } => {
            assert_eq!(fd, 1);
            assert_eq!(buf, 0x1000);
            assert_eq!(count, 5);
        }
        other => panic!("expected Write, got {:?}", other),
    }
}

#[test]
fn from_x86_64_read() {
    let syscall = LinuxSyscall::from_x86_64(0, [3, 0x2000, 128, 0, 0, 0]);
    match syscall {
        LinuxSyscall::Read { fd, buf, count } => {
            assert_eq!(fd, 3);
            assert_eq!(buf, 0x2000);
            assert_eq!(count, 128);
        }
        other => panic!("expected Read, got {:?}", other),
    }
}

#[test]
fn from_x86_64_exit_group() {
    let syscall = LinuxSyscall::from_x86_64(231, [42, 0, 0, 0, 0, 0]);
    match syscall {
        LinuxSyscall::ExitGroup { code } => assert_eq!(code, 42),
        other => panic!("expected ExitGroup, got {:?}", other),
    }
}

#[test]
fn from_x86_64_unknown() {
    let syscall = LinuxSyscall::from_x86_64(9999, [0; 6]);
    match syscall {
        LinuxSyscall::Unknown { nr } => assert_eq!(nr, 9999),
        other => panic!("expected Unknown, got {:?}", other),
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os -- from_x86_64`
Expected: FAIL — `LinuxSyscall` type does not exist yet.

**Step 3: Write the LinuxSyscall enum and from_x86_64**

Add this after the `ipc_err_to_errno` and `vm_err_to_errno` functions (around line 56), before the `SyscallBackend` trait:

```rust
// ── LinuxSyscall — CPU-agnostic syscall representation ──────────

/// CPU-agnostic Linux syscall. Each architecture maps its native
/// syscall numbers into this enum before the Linuxulator dispatches.
#[derive(Debug)]
pub enum LinuxSyscall {
    Read { fd: i32, buf: u64, count: u64 },
    Write { fd: i32, buf: u64, count: u64 },
    Close { fd: i32 },
    Fstat { fd: i32, buf: u64 },
    Mmap { addr: u64, len: u64, prot: i32, flags: i32, fd: i32, offset: u64 },
    Mprotect { addr: u64, len: u64, prot: i32 },
    Munmap { addr: u64, len: u64 },
    Brk { addr: u64 },
    RtSigaction,
    RtSigprocmask,
    Ioctl { fd: i32, request: u64 },
    Exit { code: i32 },
    ArchPrctl { code: i32, addr: u64 },
    SetTidAddress,
    ExitGroup { code: i32 },
    Openat { dirfd: i32, pathname: u64, flags: i32 },
    SetRobustList,
    Prlimit64 { pid: i32, resource: i32, new_limit: u64, old_limit_buf: u64 },
    Rseq,
    Unknown { nr: u64 },
}

impl LinuxSyscall {
    /// Map x86_64 Linux syscall numbers to `LinuxSyscall`.
    pub fn from_x86_64(nr: u64, args: [u64; 6]) -> Self {
        match nr {
            0 => LinuxSyscall::Read { fd: args[0] as i32, buf: args[1], count: args[2] },
            1 => LinuxSyscall::Write { fd: args[0] as i32, buf: args[1], count: args[2] },
            3 => LinuxSyscall::Close { fd: args[0] as i32 },
            5 => LinuxSyscall::Fstat { fd: args[0] as i32, buf: args[1] },
            9 => LinuxSyscall::Mmap {
                addr: args[0], len: args[1], prot: args[2] as i32,
                flags: args[3] as i32, fd: args[4] as i32, offset: args[5],
            },
            10 => LinuxSyscall::Mprotect { addr: args[0], len: args[1], prot: args[2] as i32 },
            11 => LinuxSyscall::Munmap { addr: args[0], len: args[1] },
            12 => LinuxSyscall::Brk { addr: args[0] },
            13 => LinuxSyscall::RtSigaction,
            14 => LinuxSyscall::RtSigprocmask,
            16 => LinuxSyscall::Ioctl { fd: args[0] as i32, request: args[1] },
            60 => LinuxSyscall::Exit { code: args[0] as i32 },
            158 => LinuxSyscall::ArchPrctl { code: args[0] as i32, addr: args[1] },
            218 => LinuxSyscall::SetTidAddress,
            231 => LinuxSyscall::ExitGroup { code: args[0] as i32 },
            257 => LinuxSyscall::Openat { dirfd: args[0] as i32, pathname: args[1], flags: args[2] as i32 },
            273 => LinuxSyscall::SetRobustList,
            302 => LinuxSyscall::Prlimit64 {
                pid: args[0] as i32, resource: args[1] as i32,
                new_limit: args[2], old_limit_buf: args[3],
            },
            334 => LinuxSyscall::Rseq,
            _ => LinuxSyscall::Unknown { nr },
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os -- from_x86_64`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-os/src/linuxulator.rs
git commit -m "feat(linuxulator): add LinuxSyscall enum and from_x86_64 mapping"
```

---

### Task 2: from_aarch64 mapping

Add the `from_aarch64` mapping function. Aarch64 Linux syscall numbers differ completely from x86_64.

**Files:**
- Modify: `harmony-os/crates/harmony-os/src/linuxulator.rs`

**Step 1: Write the failing tests**

Add to the test module:

```rust
#[test]
fn from_aarch64_write() {
    let syscall = LinuxSyscall::from_aarch64(64, [1, 0x1000, 5, 0, 0, 0]);
    match syscall {
        LinuxSyscall::Write { fd, buf, count } => {
            assert_eq!(fd, 1);
            assert_eq!(buf, 0x1000);
            assert_eq!(count, 5);
        }
        other => panic!("expected Write, got {:?}", other),
    }
}

#[test]
fn from_aarch64_read() {
    let syscall = LinuxSyscall::from_aarch64(63, [3, 0x2000, 128, 0, 0, 0]);
    match syscall {
        LinuxSyscall::Read { fd, buf, count } => {
            assert_eq!(fd, 3);
            assert_eq!(buf, 0x2000);
            assert_eq!(count, 128);
        }
        other => panic!("expected Read, got {:?}", other),
    }
}

#[test]
fn from_aarch64_exit_group() {
    let syscall = LinuxSyscall::from_aarch64(94, [42, 0, 0, 0, 0, 0]);
    match syscall {
        LinuxSyscall::ExitGroup { code } => assert_eq!(code, 42),
        other => panic!("expected ExitGroup, got {:?}", other),
    }
}

#[test]
fn from_aarch64_mmap() {
    let syscall = LinuxSyscall::from_aarch64(222, [0x1000, 4096, 3, 0x22, -1i64 as u64, 0]);
    match syscall {
        LinuxSyscall::Mmap { addr, len, prot, flags, fd, offset } => {
            assert_eq!(addr, 0x1000);
            assert_eq!(len, 4096);
            assert_eq!(prot, 3);
            assert_eq!(flags, 0x22);
            assert_eq!(fd, -1);
            assert_eq!(offset, 0);
        }
        other => panic!("expected Mmap, got {:?}", other),
    }
}

#[test]
fn from_aarch64_unknown() {
    let syscall = LinuxSyscall::from_aarch64(9999, [0; 6]);
    match syscall {
        LinuxSyscall::Unknown { nr } => assert_eq!(nr, 9999),
        other => panic!("expected Unknown, got {:?}", other),
    }
}

#[test]
fn from_aarch64_arch_prctl_maps_to_unknown() {
    // arch_prctl is x86_64-specific; aarch64 has no equivalent
    // Verify it does NOT exist in the aarch64 mapping
    let syscall = LinuxSyscall::from_aarch64(158, [0; 6]);
    assert!(matches!(syscall, LinuxSyscall::Unknown { nr: 158 }));
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os -- from_aarch64`
Expected: FAIL — `from_aarch64` does not exist.

**Step 3: Write from_aarch64**

Add this as a second method on `impl LinuxSyscall`, after `from_x86_64`:

```rust
    /// Map aarch64 Linux syscall numbers to `LinuxSyscall`.
    ///
    /// Reference: Linux kernel `include/uapi/asm-generic/unistd.h`
    /// (aarch64 uses the generic syscall table).
    pub fn from_aarch64(nr: u64, args: [u64; 6]) -> Self {
        match nr {
            56 => LinuxSyscall::Openat { dirfd: args[0] as i32, pathname: args[1], flags: args[2] as i32 },
            57 => LinuxSyscall::Close { fd: args[0] as i32 },
            63 => LinuxSyscall::Read { fd: args[0] as i32, buf: args[1], count: args[2] },
            64 => LinuxSyscall::Write { fd: args[0] as i32, buf: args[1], count: args[2] },
            80 => LinuxSyscall::Fstat { fd: args[0] as i32, buf: args[1] },
            93 => LinuxSyscall::Exit { code: args[0] as i32 },
            94 => LinuxSyscall::ExitGroup { code: args[0] as i32 },
            99 => LinuxSyscall::SetRobustList,
            134 => LinuxSyscall::RtSigaction,
            135 => LinuxSyscall::RtSigprocmask,
            214 => LinuxSyscall::Brk { addr: args[0] },
            215 => LinuxSyscall::Munmap { addr: args[0], len: args[1] },
            222 => LinuxSyscall::Mmap {
                addr: args[0], len: args[1], prot: args[2] as i32,
                flags: args[3] as i32, fd: args[4] as i32, offset: args[5],
            },
            226 => LinuxSyscall::Mprotect { addr: args[0], len: args[1], prot: args[2] as i32 },
            29 => LinuxSyscall::Ioctl { fd: args[0] as i32, request: args[1] },
            96 => LinuxSyscall::SetTidAddress,
            261 => LinuxSyscall::Prlimit64 {
                pid: args[0] as i32, resource: args[1] as i32,
                new_limit: args[2], old_limit_buf: args[3],
            },
            293 => LinuxSyscall::Rseq,
            // arch_prctl is x86_64-specific — no aarch64 equivalent
            _ => LinuxSyscall::Unknown { nr },
        }
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os -- from_aarch64`
Expected: 6 tests PASS

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-os/src/linuxulator.rs
git commit -m "feat(linuxulator): add from_aarch64 syscall number mapping"
```

---

### Task 3: Refactor handle_syscall to dispatch via LinuxSyscall enum

Replace the raw x86_64 number matching in `handle_syscall` with a new `dispatch_syscall` that takes `LinuxSyscall`. The old `handle_syscall` becomes a thin wrapper that calls `from_x86_64` then `dispatch_syscall`. All existing tests must continue to pass.

**Files:**
- Modify: `harmony-os/crates/harmony-os/src/linuxulator.rs`

**Step 1: Write the failing test**

Add to the test module:

```rust
#[test]
fn dispatch_write_via_enum() {
    let mut lx = Linuxulator::new(MockBackend::new());
    lx.init_stdio().unwrap();
    // Use dispatch_syscall with a LinuxSyscall::Write
    let msg = b"test";
    let result = lx.dispatch_syscall(LinuxSyscall::Write {
        fd: 1,
        buf: msg.as_ptr() as u64,
        count: msg.len() as u64,
    });
    assert_eq!(result, msg.len() as i64);
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os -- dispatch_write_via_enum`
Expected: FAIL — `dispatch_syscall` method does not exist.

**Step 3: Implement dispatch_syscall and refactor handle_syscall**

Add a new `pub fn dispatch_syscall(&mut self, syscall: LinuxSyscall) -> i64` method to `impl<B: SyscallBackend> Linuxulator<B>`. This method contains the actual dispatch logic (calling `self.sys_write(...)`, `self.sys_read(...)`, etc.) based on the enum variant. Then change `handle_syscall` to:

```rust
    pub fn handle_syscall(&mut self, nr: u64, args: [u64; 6]) -> i64 {
        let syscall = LinuxSyscall::from_x86_64(nr, args);
        self.dispatch_syscall(syscall)
    }

    /// Dispatch a CPU-agnostic `LinuxSyscall` to the appropriate handler.
    pub fn dispatch_syscall(&mut self, syscall: LinuxSyscall) -> i64 {
        match syscall {
            LinuxSyscall::Read { fd, buf, count } => {
                self.sys_read(fd, buf as usize, count as usize)
            }
            LinuxSyscall::Write { fd, buf, count } => {
                self.sys_write(fd, buf as usize, count as usize)
            }
            LinuxSyscall::Close { fd } => self.sys_close(fd),
            LinuxSyscall::Fstat { fd, buf } => self.sys_fstat(fd, buf as usize),
            LinuxSyscall::Mmap { addr, len, prot, flags, fd, offset } => {
                self.sys_mmap(addr, len, prot, flags, fd, offset)
            }
            LinuxSyscall::Mprotect { addr, len, prot } => {
                self.sys_mprotect(addr, len, prot)
            }
            LinuxSyscall::Munmap { addr, len } => self.sys_munmap(addr, len),
            LinuxSyscall::Brk { addr } => self.sys_brk(addr),
            LinuxSyscall::RtSigaction => self.sys_rt_sigaction(),
            LinuxSyscall::RtSigprocmask => self.sys_rt_sigprocmask(),
            LinuxSyscall::Ioctl { fd, request } => self.sys_ioctl(fd, request),
            LinuxSyscall::Exit { code } => self.sys_exit(code),
            LinuxSyscall::ArchPrctl { code, addr } => self.sys_arch_prctl(code, addr),
            LinuxSyscall::SetTidAddress => self.sys_set_tid_address(),
            LinuxSyscall::ExitGroup { code } => self.sys_exit_group(code),
            LinuxSyscall::Openat { dirfd, pathname, flags } => {
                self.sys_openat(dirfd, pathname as usize, flags)
            }
            LinuxSyscall::SetRobustList => self.sys_set_robust_list(),
            LinuxSyscall::Prlimit64 { pid, resource, new_limit, old_limit_buf } => {
                self.sys_prlimit64(pid, resource, new_limit, old_limit_buf as usize)
            }
            LinuxSyscall::Rseq => ENOSYS,
            LinuxSyscall::Unknown { .. } => ENOSYS,
        }
    }
```

**Step 4: Run ALL tests to verify nothing broke**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os`
Expected: ALL existing tests pass + new `dispatch_write_via_enum` test passes.

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-os/src/linuxulator.rs
git commit -m "refactor(linuxulator): dispatch via LinuxSyscall enum instead of raw numbers"
```

---

### Task 4: ELF loader — accept EM_AARCH64

Extend the ELF parser to accept `EM_AARCH64` (0xB7) on aarch64 targets. The machine check becomes `cfg`-gated so each target accepts only its own machine type.

**Files:**
- Modify: `harmony-os/crates/harmony-os/src/elf.rs`

**Step 1: Write the failing tests**

Add to the existing test module in `elf.rs`. First, add a helper to build aarch64 test ELFs:

```rust
    /// Build a minimal valid ELF64 binary with EM_AARCH64.
    fn build_test_elf_aarch64(code: &[u8]) -> Vec<u8> {
        let mut elf = build_test_elf(code);
        // Change e_machine from EM_X86_64 (0x3E) to EM_AARCH64 (0xB7)
        elf[18..20].copy_from_slice(&0xB7u16.to_le_bytes());
        elf
    }

    #[test]
    fn accept_native_machine_type() {
        let code = [0xCC; 16];
        // On x86_64 hosts, EM_X86_64 should be accepted
        #[cfg(target_arch = "x86_64")]
        {
            let elf = build_test_elf(&code);
            assert!(parse_elf(&elf).is_ok());
        }
        // On aarch64 hosts, EM_AARCH64 should be accepted
        #[cfg(target_arch = "aarch64")]
        {
            let elf = build_test_elf_aarch64(&code);
            assert!(parse_elf(&elf).is_ok());
        }
    }

    #[test]
    fn reject_foreign_machine_type() {
        let code = [0xCC; 16];
        // On x86_64 hosts, EM_AARCH64 should be rejected
        #[cfg(target_arch = "x86_64")]
        {
            let elf = build_test_elf_aarch64(&code);
            assert_eq!(parse_elf(&elf), Err(ElfError::UnsupportedMachine));
        }
        // On aarch64 hosts, EM_X86_64 should be rejected
        #[cfg(target_arch = "aarch64")]
        {
            let elf = build_test_elf(&code);
            assert_eq!(parse_elf(&elf), Err(ElfError::UnsupportedMachine));
        }
    }
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os -- reject_foreign_machine_type`
Expected: FAIL — `ElfError::UnsupportedMachine` variant doesn't exist (currently `NotX86_64`).

**Step 3: Implement the changes**

In `elf.rs`:

1. Add `EM_AARCH64` constant:
```rust
const EM_AARCH64: u16 = 0xB7;
```

2. Rename `ElfError::NotX86_64` to `ElfError::UnsupportedMachine`:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElfError {
    TooShort,
    BadMagic,
    Not64Bit,
    NotLittleEndian,
    NotExecutable,
    UnsupportedMachine,  // was NotX86_64
    InvalidPhdr,
    SegmentOutOfBounds,
}
```

3. Replace the machine check in `parse_elf` (around line 137-139):
```rust
    // Must be the native machine type
    let e_machine = u16_le(data, 18);
    #[cfg(target_arch = "x86_64")]
    let expected_machine = EM_X86_64;
    #[cfg(target_arch = "aarch64")]
    let expected_machine = EM_AARCH64;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    return Err(ElfError::UnsupportedMachine);

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if e_machine != expected_machine {
        return Err(ElfError::UnsupportedMachine);
    }
```

4. Update the existing `reject_non_x86_64` test to use the new error variant:
```rust
    #[test]
    fn reject_non_x86_64() {
        let mut elf = build_test_elf(&[0xCC]);
        elf[18..20].copy_from_slice(&0x03u16.to_le_bytes()); // EM_386
        assert_eq!(parse_elf(&elf), Err(ElfError::UnsupportedMachine));
    }
```

5. Also update any other code that references `ElfError::NotX86_64` (grep for it — the boot crate may reference it).

**Step 4: Run ALL tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os`
Expected: All tests pass including the new and updated ones.

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-os/src/elf.rs
git commit -m "feat(elf): accept EM_AARCH64, rename NotX86_64 to UnsupportedMachine"
```

---

### Task 5: TrapFrame struct (harmony-boot-aarch64)

Create `syscall.rs` in the aarch64 boot crate with the `TrapFrame` struct and layout tests. This is pure data — no asm, no hardware access.

**Files:**
- Create: `harmony-os/crates/harmony-boot-aarch64/src/syscall.rs`
- Modify: `harmony-os/crates/harmony-boot-aarch64/src/main.rs` (add `mod syscall;`)

**Step 1: Write the failing tests**

Create `crates/harmony-boot-aarch64/src/syscall.rs`:

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! Aarch64 syscall trap handler — TrapFrame and SVC dispatch.
//!
//! The exception vector table (in `vectors.rs`) saves registers into a
//! `TrapFrame` and calls into the Rust handlers defined here.

// Hardware functions are only compiled for aarch64; suppress warnings on
// the host test runner.
#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

/// Saved register state on exception entry.
///
/// The assembly vector table saves X0-X30, ELR_EL1, and SPSR_EL1 in
/// this exact layout. The struct must be `#[repr(C)]` so field offsets
/// match the assembly push order.
#[repr(C)]
pub struct TrapFrame {
    /// General-purpose registers X0-X30.
    pub x: [u64; 31],
    /// Exception Link Register — the PC to return to.
    pub elr: u64,
    /// Saved Processor State Register.
    pub spsr: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem;

    #[test]
    fn trap_frame_size() {
        // 31 registers * 8 bytes + ELR (8) + SPSR (8) = 264 bytes
        assert_eq!(mem::size_of::<TrapFrame>(), 264);
    }

    #[test]
    fn trap_frame_x8_offset() {
        // X8 (syscall number) is at index 8 in the x array
        // Offset = 8 * 8 = 64 bytes from the start of the struct
        assert_eq!(mem::offset_of!(TrapFrame, x) + 8 * 8, 64);
    }

    #[test]
    fn trap_frame_elr_offset() {
        // ELR comes after 31 u64s = 248 bytes
        assert_eq!(mem::offset_of!(TrapFrame, elr), 248);
    }

    #[test]
    fn trap_frame_spsr_offset() {
        // SPSR comes after ELR = 256 bytes
        assert_eq!(mem::offset_of!(TrapFrame, spsr), 256);
    }
}
```

**Step 2: Add the module to main.rs**

In `crates/harmony-boot-aarch64/src/main.rs`, add after the other `mod` declarations (around line 14):

```rust
mod syscall;
```

**Step 3: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os/crates/harmony-boot-aarch64 && cargo test --target $(rustc -vV | grep host | cut -d' ' -f2) -- trap_frame`
Expected: 4 tests PASS (these are pure layout tests, not TDD red-green since the struct and tests are written together).

**Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-boot-aarch64/src/syscall.rs crates/harmony-boot-aarch64/src/main.rs
git commit -m "feat(boot-aarch64): add TrapFrame struct with layout tests"
```

---

### Task 6: Exception vector table (harmony-boot-aarch64)

Create `vectors.rs` with the VBAR_EL1 vector table in assembly. This is aarch64-only — no host tests for the asm, but we add an `init` function.

**Files:**
- Create: `harmony-os/crates/harmony-boot-aarch64/src/vectors.rs`
- Modify: `harmony-os/crates/harmony-boot-aarch64/src/main.rs` (add `mod vectors;`)

**Step 1: Create vectors.rs**

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! Aarch64 exception vector table (VBAR_EL1).
//!
//! Provides a 2048-byte aligned vector table with 16 entries. Only the
//! "Current EL with SPx, Synchronous" entry (offset 0x200) has real
//! dispatch logic — it reads ESR_EL1 to determine the exception class
//! and branches to the appropriate Rust handler. All other entries halt
//! with an unexpected-exception panic.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code, unused_imports))]

/// Install the exception vector table.
///
/// Writes VBAR_EL1 to point to our vector table and issues an ISB to
/// ensure the new value takes effect before the next exception.
///
/// # Safety
/// Must be called once, at EL1, before any SVC or fault can occur.
#[cfg(target_arch = "aarch64")]
pub unsafe fn init() {
    core::arch::asm!(
        "adr {tmp}, vector_table",
        "msr vbar_el1, {tmp}",
        "isb",
        tmp = out(reg) _,
    );
}

/// The exception vector table and its entries.
///
/// Each entry is exactly 128 bytes (0x80). The table is 2048 bytes
/// total (16 entries) and must be 2048-byte aligned.
///
/// Layout:
///   0x000  Current EL, SP_EL0, Synchronous   — unused (we use SPx)
///   0x080  Current EL, SP_EL0, IRQ           — unused
///   0x100  Current EL, SP_EL0, FIQ           — unused
///   0x180  Current EL, SP_EL0, SError        — unused
///   0x200  Current EL, SPx, Synchronous      — SVC + abort dispatch
///   0x280  Current EL, SPx, IRQ              — unexpected
///   0x300  Current EL, SPx, FIQ              — unexpected
///   0x380  Current EL, SPx, SError           — unexpected
///   0x400  Lower EL, AArch64, Synchronous    — unused (no EL0)
///   0x480  Lower EL, AArch64, IRQ            — unused
///   0x500  Lower EL, AArch64, FIQ            — unused
///   0x580  Lower EL, AArch64, SError         — unused
///   0x600  Lower EL, AArch32, Synchronous    — unused
///   0x680  Lower EL, AArch32, IRQ            — unused
///   0x700  Lower EL, AArch32, FIQ            — unused
///   0x780  Lower EL, AArch32, SError         — unused
#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    // ── Vector table ────────────────────────────────────────────
    ".balign 2048",
    "vector_table:",

    // 0x000 — Current EL, SP_EL0, Synchronous (unused)
    "b unexpected_exception",
    ".balign 0x80",

    // 0x080 — Current EL, SP_EL0, IRQ (unused)
    "b unexpected_exception",
    ".balign 0x80",

    // 0x100 — Current EL, SP_EL0, FIQ (unused)
    "b unexpected_exception",
    ".balign 0x80",

    // 0x180 — Current EL, SP_EL0, SError (unused)
    "b unexpected_exception",
    ".balign 0x80",

    // ── 0x200 — Current EL, SPx, Synchronous ── THE MAIN ENTRY ──
    //
    // Save all general-purpose registers + ELR + SPSR to the stack
    // (TrapFrame layout: x[31], elr, spsr = 264 bytes).
    // Then read ESR_EL1 to determine exception class and dispatch.

    // Allocate TrapFrame on the stack (264 bytes, rounded up to 272 for 16-byte alignment)
    "sub sp, sp, #272",

    // Save X0-X29 as pairs
    "stp x0,  x1,  [sp, #0]",
    "stp x2,  x3,  [sp, #16]",
    "stp x4,  x5,  [sp, #32]",
    "stp x6,  x7,  [sp, #48]",
    "stp x8,  x9,  [sp, #64]",
    "stp x10, x11, [sp, #80]",
    "stp x12, x13, [sp, #96]",
    "stp x14, x15, [sp, #112]",
    "stp x16, x17, [sp, #128]",
    "stp x18, x19, [sp, #144]",
    "stp x20, x21, [sp, #160]",
    "stp x22, x23, [sp, #176]",
    "stp x24, x25, [sp, #192]",
    "stp x26, x27, [sp, #208]",
    "stp x28, x29, [sp, #224]",
    // Save X30 (LR)
    "str x30, [sp, #240]",
    // Save ELR_EL1 and SPSR_EL1
    "mrs x10, elr_el1",
    "mrs x11, spsr_el1",
    "stp x10, x11, [sp, #248]",

    // Read ESR_EL1 for exception class
    "mrs x1, esr_el1",
    // EC = ESR[31:26]
    "lsr x2, x1, #26",

    // EC == 0x15 → SVC from AArch64
    "cmp x2, #0x15",
    "b.eq call_svc_handler",

    // EC == 0x25 → Data Abort, current EL
    "cmp x2, #0x25",
    "b.eq call_abort_handler",

    // EC == 0x21 → Instruction Abort, current EL
    "cmp x2, #0x21",
    "b.eq call_abort_handler",

    // Other — fall through to abort handler with ESR in x1
    "b call_abort_handler",

    // ── SVC handler dispatch ──
    "call_svc_handler:",
    "mov x0, sp",              // x0 = &TrapFrame
    "bl svc_handler",          // svc_handler(&mut TrapFrame)

    // Restore ELR and SPSR
    "ldp x10, x11, [sp, #248]",
    "msr elr_el1, x10",
    "msr spsr_el1, x11",

    // Restore X0-X29
    "ldp x0,  x1,  [sp, #0]",
    "ldp x2,  x3,  [sp, #16]",
    "ldp x4,  x5,  [sp, #32]",
    "ldp x6,  x7,  [sp, #48]",
    "ldp x8,  x9,  [sp, #64]",
    "ldp x10, x11, [sp, #80]",
    "ldp x12, x13, [sp, #96]",
    "ldp x14, x15, [sp, #112]",
    "ldp x16, x17, [sp, #128]",
    "ldp x18, x19, [sp, #144]",
    "ldp x20, x21, [sp, #160]",
    "ldp x22, x23, [sp, #176]",
    "ldp x24, x25, [sp, #192]",
    "ldp x26, x27, [sp, #208]",
    "ldp x28, x29, [sp, #224]",
    "ldr x30, [sp, #240]",

    // Deallocate TrapFrame
    "add sp, sp, #272",
    "eret",

    ".balign 0x80",

    // ── Abort handler dispatch ──
    "call_abort_handler:",
    "mov x0, sp",              // x0 = &TrapFrame
    // x1 already contains ESR_EL1 from above
    "bl abort_handler",        // abort_handler(&TrapFrame, esr: u64) -> !
    "b unexpected_exception",  // should not return

    // 0x280 — Current EL, SPx, IRQ (unexpected)
    ".balign 0x80",
    "b unexpected_exception",

    // 0x300 — Current EL, SPx, FIQ (unexpected)
    ".balign 0x80",
    "b unexpected_exception",

    // 0x380 — Current EL, SPx, SError (unexpected)
    ".balign 0x80",
    "b unexpected_exception",

    // 0x400-0x780 — Lower EL entries (8 entries, all unused)
    ".balign 0x80",
    "b unexpected_exception",
    ".balign 0x80",
    "b unexpected_exception",
    ".balign 0x80",
    "b unexpected_exception",
    ".balign 0x80",
    "b unexpected_exception",
    ".balign 0x80",
    "b unexpected_exception",
    ".balign 0x80",
    "b unexpected_exception",
    ".balign 0x80",
    "b unexpected_exception",
    ".balign 0x80",
    "b unexpected_exception",

    // ── Unexpected exception handler ────────────────────────────
    "unexpected_exception:",
    "b unexpected_exception",  // infinite loop — panic will be added later
);
```

**Step 2: Add the module to main.rs**

In `crates/harmony-boot-aarch64/src/main.rs`, add after the other `mod` declarations:

```rust
mod vectors;
```

**Step 3: Run tests to verify compilation**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os/crates/harmony-boot-aarch64 && cargo test --target $(rustc -vV | grep host | cut -d' ' -f2)`
Expected: All existing tests pass. The `global_asm!` is `cfg(target_arch = "aarch64")` so it's not compiled on the host — just verify the module compiles without errors.

**Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-boot-aarch64/src/vectors.rs crates/harmony-boot-aarch64/src/main.rs
git commit -m "feat(boot-aarch64): add exception vector table with SVC + abort dispatch"
```

---

### Task 7: SVC handler and abort handler (harmony-boot-aarch64)

Add the Rust-side handlers that the asm vector table calls into. The SVC handler extracts syscall args from the TrapFrame and calls the Linuxulator. The abort handler prints diagnostics and panics.

**Files:**
- Modify: `harmony-os/crates/harmony-boot-aarch64/src/syscall.rs`
- Modify: `harmony-os/crates/harmony-boot-aarch64/Cargo.toml` (add harmony-os dependency)

**Step 1: Add harmony-os dependency to boot-aarch64**

In `crates/harmony-boot-aarch64/Cargo.toml`, add under `[dependencies]`:

```toml
harmony-os = { path = "../harmony-os", default-features = false }
```

**Step 2: Implement the handlers**

Add to `crates/harmony-boot-aarch64/src/syscall.rs`, after the TrapFrame struct:

```rust
use harmony_os::linuxulator::{Linuxulator, LinuxSyscall, SyscallBackend};

/// Global dispatch function pointer. Set during boot before SVC is possible.
static mut DISPATCH_FN: Option<fn(LinuxSyscall) -> SyscallDispatchResult> = None;

/// Result from syscall dispatch, matching x86_64's pattern.
pub struct SyscallDispatchResult {
    pub retval: i64,
    pub exited: bool,
    pub exit_code: i32,
}

/// Install the syscall dispatch function.
///
/// # Safety
/// Must be called before the vector table is installed and before any
/// SVC instruction executes.
pub unsafe fn set_dispatch_fn(f: fn(LinuxSyscall) -> SyscallDispatchResult) {
    DISPATCH_FN = Some(f);
}

/// Check if the process has exited.
pub fn process_exited() -> bool {
    unsafe { PROCESS_EXITED }
}

/// Get the exit code.
pub fn exit_code() -> i32 {
    unsafe { EXIT_CODE }
}

static mut PROCESS_EXITED: bool = false;
static mut EXIT_CODE: i32 = 0;

/// Rust SVC handler — called from the exception vector table asm.
///
/// Reads the syscall number from X8 and arguments from X0-X5,
/// maps to a `LinuxSyscall` via `from_aarch64`, dispatches, and
/// writes the return value back to X0 in the TrapFrame.
#[cfg(target_arch = "aarch64")]
#[no_mangle]
pub unsafe extern "C" fn svc_handler(frame: &mut TrapFrame) {
    let nr = frame.x[8];
    let args = [
        frame.x[0], frame.x[1], frame.x[2],
        frame.x[3], frame.x[4], frame.x[5],
    ];

    let syscall = LinuxSyscall::from_aarch64(nr, args);

    if let Some(dispatch) = DISPATCH_FN {
        let result = dispatch(syscall);
        if result.exited {
            PROCESS_EXITED = true;
            EXIT_CODE = result.exit_code;
            // Halt — do not return to the binary.
            loop {
                core::arch::asm!("wfe");
            }
        }
        frame.x[0] = result.retval as u64;
    } else {
        // No dispatch function installed — return ENOSYS
        frame.x[0] = (-38i64) as u64;
    }
}

/// Rust abort handler — called from the exception vector table asm.
///
/// Prints diagnostic info (faulting address, PC, syndrome) to PL011
/// serial and halts via panic.
#[cfg(target_arch = "aarch64")]
#[no_mangle]
pub unsafe extern "C" fn abort_handler(frame: &TrapFrame, esr: u64) -> ! {
    let ec = (esr >> 26) & 0x3F;
    let iss = esr & 0x1FF_FFFF;
    let far: u64;
    core::arch::asm!("mrs {}, far_el1", out(reg) far);

    let kind = match ec {
        0x21 => "Instruction Abort (current EL)",
        0x25 => "Data Abort (current EL)",
        _ => "Unhandled Synchronous Exception",
    };

    panic!(
        "{} at ELR={:#x} FAR={:#x} ESR={:#x} (EC={:#x} ISS={:#x})",
        kind, frame.elr, far, esr, ec, iss
    );
}
```

**Step 3: Run tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os/crates/harmony-boot-aarch64 && cargo test --target $(rustc -vV | grep host | cut -d' ' -f2)`
Expected: All existing tests pass. The handlers are `cfg(target_arch = "aarch64")` — on the host, only the TrapFrame tests run.

**Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-boot-aarch64/src/syscall.rs crates/harmony-boot-aarch64/Cargo.toml
git commit -m "feat(boot-aarch64): add SVC handler + abort handler for exception dispatch"
```

---

### Task 8: Boot integration — wire up vectors + Linuxulator

Connect the exception vector table and Linuxulator into the aarch64 boot sequence in `main.rs`. This adds the vector table init, Linuxulator setup, and optional ELF binary execution.

**Files:**
- Modify: `harmony-os/crates/harmony-boot-aarch64/src/main.rs`

**Step 1: Add the integration code**

After the existing `[Runtime]` creation (around line 252), before the idle loop, add:

```rust
    // ── Install exception vector table ──
    unsafe { vectors::init() };
    let _ = writeln!(serial, "[Vectors] Exception vector table installed");

    // ── Initialise Linuxulator ──
    // Create a SyscallBackend using the kernel's 9P file server.
    // For the MVP, we use the same serial-backed approach as x86_64.
    use harmony_os::linuxulator::{Linuxulator, LinuxSyscall};

    static mut LINUXULATOR: Option<Linuxulator</* backend type */>> = None;
    // ... backend setup matches x86_64 pattern ...

    // Install dispatch function
    fn dispatch(syscall: LinuxSyscall) -> syscall::SyscallDispatchResult {
        let lx = unsafe { LINUXULATOR.as_mut().unwrap() };
        let retval = lx.dispatch_syscall(syscall);
        syscall::SyscallDispatchResult {
            retval,
            exited: lx.exited(),
            exit_code: lx.exit_code().unwrap_or(0),
        }
    }
    unsafe { syscall::set_dispatch_fn(dispatch) };
```

**Note:** The exact backend type depends on what `SyscallBackend` implementation exists for aarch64. If none exists yet, create a minimal serial-backed one that maps stdout/stderr writes to PL011 serial output — matching what x86_64 does. This integration is highly dependent on what infrastructure exists; the implementer should reference `harmony-boot/src/main.rs` lines 500-590 for the x86_64 pattern.

**Step 2: Run tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os/crates/harmony-boot-aarch64 && cargo test --target $(rustc -vV | grep host | cut -d' ' -f2)`
Expected: All tests pass. The new integration code is `cfg(target_os = "uefi")` gated.

Also run workspace tests:
Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test --workspace`
Expected: All tests pass.

**Step 3: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-boot-aarch64/src/main.rs
git commit -m "feat(boot-aarch64): wire up exception vectors + Linuxulator dispatch"
```

---

### Task 9: Final verification

Run all tests across both crates and verify clippy is clean.

**Step 1: Run all harmony-os workspace tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test --workspace`
Expected: All tests pass.

**Step 2: Run boot-aarch64 tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os/crates/harmony-boot-aarch64 && cargo test --target $(rustc -vV | grep host | cut -d' ' -f2)`
Expected: All tests pass (including new TrapFrame layout tests).

**Step 3: Run clippy**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo clippy --workspace`
Expected: Zero warnings.

**Step 4: Final commit if any cleanup needed**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add -A
git commit -m "chore: final cleanup for aarch64 syscall trap"
```

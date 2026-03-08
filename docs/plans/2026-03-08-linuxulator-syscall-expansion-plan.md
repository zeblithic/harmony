# Linuxulator Syscall Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand Linuxulator from 2 syscalls to ~18, enabling a musl-static hello world to run on QEMU.

**Architecture:** Monolithic expansion of `linuxulator.rs` — add `MemoryArena` struct, ~16 new `sys_*` methods, `ipc_err_to_errno` helper. Add `stat` to `SyscallBackend` trait. Add `write_fs_base`/`read_fs_base` to `syscall.rs`. All in harmony-os + harmony-boot crates.

**Tech Stack:** Rust, no_std, x86_64 inline asm (for FS base), harmony-microkernel (9P IPC)

---

## Background for the Implementer

### What exists today

- `crates/harmony-os/src/linuxulator.rs` (~450 lines) — `SyscallBackend` trait (walk/open/read/write/clunk), `MockBackend` for tests, `Linuxulator<B>` struct with fd table, `handle_syscall` dispatching `write` (nr=1) and `exit_group` (nr=231).
- `crates/harmony-boot/src/syscall.rs` (~213 lines) — x86_64 MSR setup, naked asm syscall entry point, `SyscallDispatchFn` callback.
- `crates/harmony-boot/src/main.rs` (~619 lines) — Ring 3 block with `DirectBackend`, ELF loading, Linuxulator init, jump to binary.

### Key types from harmony-microkernel

```rust
pub type Fid = u32;
pub type QPath = u64;
pub enum OpenMode { Read, Write, ReadWrite }
pub enum FileType { Regular, Directory }
pub struct FileStat { pub qpath: QPath, pub name: Arc<str>, pub size: u64, pub file_type: FileType }
pub enum IpcError { NotFound, PermissionDenied, NotOpen, InvalidFid, NotDirectory, IsDirectory, ReadOnly, ResourceExhausted, Conflict, NotSupported, InvalidArgument }
```

### Linux x86_64 syscall numbers

```
0=read, 1=write, 3=close, 5=fstat, 9=mmap, 11=munmap, 12=brk,
13=rt_sigaction, 14=rt_sigprocmask, 16=ioctl, 60=exit, 158=arch_prctl,
218=set_tid_address, 231=exit_group, 257=openat, 273=set_robust_list,
302=prlimit64, 334=rseq
```

### Design doc

Full design: `docs/plans/2026-03-08-linuxulator-syscall-expansion-design.md`

---

### Task 1: Add `stat` to SyscallBackend and expand errno constants

**Files:**
- Modify: `crates/harmony-os/src/linuxulator.rs:13` (imports), `crates/harmony-os/src/linuxulator.rs:15-19` (errno constants), `crates/harmony-os/src/linuxulator.rs:26-32` (SyscallBackend trait), `crates/harmony-os/src/linuxulator.rs:38-86` (MockBackend)

**Step 1: Write the failing test**

Add to the `tests` module at the bottom of `linuxulator.rs`:

```rust
#[test]
fn ipc_err_to_errno_maps_all_variants() {
    assert_eq!(ipc_err_to_errno(IpcError::NotFound), -2);
    assert_eq!(ipc_err_to_errno(IpcError::PermissionDenied), -13);
    assert_eq!(ipc_err_to_errno(IpcError::NotOpen), -9);
    assert_eq!(ipc_err_to_errno(IpcError::InvalidFid), -9);
    assert_eq!(ipc_err_to_errno(IpcError::NotDirectory), -20);
    assert_eq!(ipc_err_to_errno(IpcError::IsDirectory), -21);
    assert_eq!(ipc_err_to_errno(IpcError::ReadOnly), -30);
    assert_eq!(ipc_err_to_errno(IpcError::ResourceExhausted), -12);
    assert_eq!(ipc_err_to_errno(IpcError::Conflict), -17);
    assert_eq!(ipc_err_to_errno(IpcError::NotSupported), -38);
    assert_eq!(ipc_err_to_errno(IpcError::InvalidArgument), -22);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-os ipc_err_to_errno_maps_all_variants`
Expected: FAIL — `ipc_err_to_errno` doesn't exist yet

**Step 3: Implement**

1. Update the import line (line 13) to also import `FileStat, FileType`:
```rust
use harmony_microkernel::{Fid, FileStat, FileType, IpcError, OpenMode, QPath};
```

2. Expand the errno constants section (after line 19):
```rust
const EBADF: i64 = -9;
const EIO: i64 = -5;
const ENOSYS: i64 = -38;
const ENOMEM: i64 = -12;
const EINVAL: i64 = -22;
const ENOTTY: i64 = -25;
const ESRCH: i64 = -3;
```

3. Add `stat` to the `SyscallBackend` trait (after `clunk`):
```rust
fn stat(&mut self, fid: Fid) -> Result<FileStat, IpcError>;
```

4. Add `stat` to `MockBackend`:
   - Add field: `pub stats: Vec<Fid>,`
   - Initialize in `new()`: `stats: Vec::new(),`
   - Implement:
```rust
fn stat(&mut self, fid: Fid) -> Result<FileStat, IpcError> {
    self.stats.push(fid);
    Ok(FileStat {
        qpath: 0,
        name: alloc::sync::Arc::from("mock"),
        size: 0,
        file_type: FileType::Regular,
    })
}
```

5. Add the `ipc_err_to_errno` function (after the errno constants, before SyscallBackend):
```rust
fn ipc_err_to_errno(e: IpcError) -> i64 {
    match e {
        IpcError::NotFound => -2,           // ENOENT
        IpcError::PermissionDenied => -13,  // EACCES
        IpcError::NotOpen => -9,            // EBADF
        IpcError::InvalidFid => -9,         // EBADF
        IpcError::NotDirectory => -20,      // ENOTDIR
        IpcError::IsDirectory => -21,       // EISDIR
        IpcError::ReadOnly => -30,          // EROFS
        IpcError::ResourceExhausted => -12, // ENOMEM
        IpcError::Conflict => -17,          // EEXIST
        IpcError::NotSupported => -38,      // ENOSYS
        IpcError::InvalidArgument => -22,   // EINVAL
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-os`
Expected: ALL tests pass (existing + new)

**Step 5: Commit**

```bash
git add crates/harmony-os/src/linuxulator.rs
git commit -m "feat(linuxulator): add stat to SyscallBackend, ipc_err_to_errno, expand errno constants"
```

---

### Task 2: MemoryArena — brk and mmap

**Files:**
- Modify: `crates/harmony-os/src/linuxulator.rs` (add MemoryArena struct, add arena field to Linuxulator)

**Step 1: Write failing tests**

Add to the `tests` module:

```rust
#[test]
fn arena_brk_probe_returns_base() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::with_arena(mock, 64 * 1024); // 64 KiB arena
    let base = lx.handle_syscall(12, [0, 0, 0, 0, 0, 0]); // brk(0) = probe
    assert!(base > 0);
    assert_eq!(base as usize, lx.arena_base());
}

#[test]
fn arena_brk_extend_returns_new_brk() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::with_arena(mock, 64 * 1024);
    let base = lx.handle_syscall(12, [0, 0, 0, 0, 0, 0]) as u64;
    // Request brk at base + 8192 (2 pages)
    let new_brk = lx.handle_syscall(12, [base + 8192, 0, 0, 0, 0, 0]);
    assert_eq!(new_brk as u64, base + 8192);
}

#[test]
fn arena_brk_aligns_to_4k() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::with_arena(mock, 64 * 1024);
    let base = lx.handle_syscall(12, [0, 0, 0, 0, 0, 0]) as u64;
    // Request non-aligned brk
    let new_brk = lx.handle_syscall(12, [base + 100, 0, 0, 0, 0, 0]);
    assert_eq!(new_brk as u64, base + 4096); // rounded up to 4K
}

#[test]
fn arena_mmap_anonymous_returns_valid_address() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::with_arena(mock, 64 * 1024);
    let base = lx.arena_base();
    let arena_size = 64 * 1024;
    // mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
    let addr = lx.handle_syscall(9, [0, 4096, 3, 0x22, u64::MAX, 0]);
    assert!(addr > 0); // not an error
    let addr = addr as usize;
    // Should be within the arena, near the top
    assert!(addr >= base);
    assert!(addr < base + arena_size);
    // Should be 4K-aligned
    assert_eq!(addr % 4096, 0);
}

#[test]
fn arena_mmap_is_zero_filled() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::with_arena(mock, 64 * 1024);
    let addr = lx.handle_syscall(9, [0, 4096, 3, 0x22, u64::MAX, 0]) as usize;
    // Check that memory is zeroed
    let slice = unsafe { core::slice::from_raw_parts(addr as *const u8, 4096) };
    assert!(slice.iter().all(|&b| b == 0));
}

#[test]
fn arena_brk_cannot_exceed_mmap() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::with_arena(mock, 16 * 1024); // small: 16 KiB
    let base = lx.handle_syscall(12, [0, 0, 0, 0, 0, 0]) as u64;
    // mmap 8 KiB from the top
    let _addr = lx.handle_syscall(9, [0, 8192, 3, 0x22, u64::MAX, 0]);
    // Try to brk past the mmap region — should fail (return current brk)
    let result = lx.handle_syscall(12, [base + 16384, 0, 0, 0, 0, 0]) as u64;
    assert!(result < base + 16384); // didn't get what we asked for
}

#[test]
fn arena_mmap_exhaustion_returns_enomem() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::with_arena(mock, 16 * 1024); // 16 KiB
    // Try to mmap more than the arena
    let result = lx.handle_syscall(9, [0, 32768, 3, 0x22, u64::MAX, 0]);
    assert_eq!(result, ENOMEM);
}

#[test]
fn arena_munmap_returns_success() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::with_arena(mock, 64 * 1024);
    let addr = lx.handle_syscall(9, [0, 4096, 3, 0x22, u64::MAX, 0]) as u64;
    let result = lx.handle_syscall(11, [addr, 4096, 0, 0, 0, 0]);
    assert_eq!(result, 0);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-os arena_`
Expected: FAIL — `Linuxulator::with_arena` and `arena_base` don't exist

**Step 3: Implement**

1. Add `MemoryArena` struct (before the Linuxulator struct):
```rust
// ── Memory arena ──────────────────────────────────────────────────

const PAGE_SIZE: usize = 4096;

/// A 4KB-page-aligned memory arena for the Linux process.
///
/// brk grows up from offset 0, mmap grows down from the top.
/// All allocations are 4KB-aligned. Bump-only — munmap is a no-op.
struct MemoryArena {
    pages: Vec<u8>,
    base: usize,
    brk_offset: usize,
    mmap_regions: Vec<(usize, usize)>,
    mmap_top: usize,
}

impl MemoryArena {
    fn new(size: usize) -> Self {
        let size = (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1); // round up
        let pages = alloc::vec![0u8; size];
        let base = pages.as_ptr() as usize;
        Self {
            pages,
            base,
            brk_offset: 0,
            mmap_regions: Vec::new(),
            mmap_top: size,
        }
    }

    fn size(&self) -> usize {
        self.pages.len()
    }
}
```

2. Add `arena` field to `Linuxulator`:
```rust
pub struct Linuxulator<B: SyscallBackend> {
    backend: B,
    fd_table: BTreeMap<i32, Fid>,
    next_fid: Fid,
    exit_code: Option<i32>,
    arena: MemoryArena,
}
```

3. Update `Linuxulator::new` to create a default 1 MiB arena, add `with_arena`:
```rust
pub fn new(backend: B) -> Self {
    Self::with_arena(backend, 1024 * 1024) // 1 MiB default
}

pub fn with_arena(backend: B, arena_size: usize) -> Self {
    Self {
        backend,
        fd_table: BTreeMap::new(),
        next_fid: 100,
        exit_code: None,
        arena: MemoryArena::new(arena_size),
    }
}
```

4. Add `arena_base` accessor (for tests):
```rust
#[cfg(test)]
pub fn arena_base(&self) -> usize {
    self.arena.base
}
```

5. Add `sys_brk`, `sys_mmap`, `sys_munmap` methods and wire them into `handle_syscall`:

```rust
fn sys_brk(&mut self, addr: u64) -> i64 {
    let base = self.arena.base as u64;
    if addr == 0 {
        return (base + self.arena.brk_offset as u64) as i64;
    }
    if addr < base {
        return (base + self.arena.brk_offset as u64) as i64;
    }
    let requested_offset = (addr - base) as usize;
    if requested_offset > self.arena.mmap_top {
        return (base + self.arena.brk_offset as u64) as i64;
    }
    self.arena.brk_offset = (requested_offset + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    (base + self.arena.brk_offset as u64) as i64
}

fn sys_mmap(&mut self, _addr: u64, length: u64, _prot: i32, flags: i32, _fd: i32, _offset: u64) -> i64 {
    let len = ((length as usize) + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let map_anonymous = 0x20;
    if flags & map_anonymous == 0 {
        return ENOSYS;
    }
    if len > self.arena.mmap_top.saturating_sub(self.arena.brk_offset) {
        return ENOMEM;
    }
    self.arena.mmap_top -= len;
    let ptr = self.arena.base + self.arena.mmap_top;
    unsafe { core::ptr::write_bytes(ptr as *mut u8, 0, len); }
    self.arena.mmap_regions.push((self.arena.mmap_top, len));
    ptr as i64
}

fn sys_munmap(&mut self, _addr: u64, _length: u64) -> i64 {
    0 // no-op in bump allocator
}
```

6. Update `handle_syscall` match:
```rust
pub fn handle_syscall(&mut self, nr: u64, args: [u64; 6]) -> i64 {
    match nr {
        1 => self.sys_write(args[0] as i32, args[1] as usize, args[2] as usize),
        9 => self.sys_mmap(args[0], args[1], args[2] as i32, args[3] as i32, args[4] as i32, args[5]),
        11 => self.sys_munmap(args[0], args[1]),
        12 => self.sys_brk(args[0]),
        231 => self.sys_exit_group(args[0] as i32),
        _ => ENOSYS,
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-os`
Expected: ALL tests pass

**Step 5: Commit**

```bash
git add crates/harmony-os/src/linuxulator.rs
git commit -m "feat(linuxulator): MemoryArena with 4KB pages — brk, mmap, munmap"
```

---

### Task 3: File descriptor operations — read, close, openat

**Files:**
- Modify: `crates/harmony-os/src/linuxulator.rs`

**Step 1: Write failing tests**

Add to the `tests` module:

```rust
#[test]
fn sys_read_copies_data_to_buffer() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    lx.init_stdio().unwrap();

    // Read from stdin (fd 0) — MockBackend returns empty by default
    let mut buf = [0xFFu8; 64];
    let result = lx.handle_syscall(0, [0, buf.as_mut_ptr() as u64, 64, 0, 0, 0]);
    assert_eq!(result, 0); // MockBackend returns empty Vec
}

#[test]
fn sys_read_bad_fd() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    let mut buf = [0u8; 64];
    let result = lx.handle_syscall(0, [99, buf.as_mut_ptr() as u64, 64, 0, 0, 0]);
    assert_eq!(result, EBADF);
}

#[test]
fn sys_read_zero_count() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    lx.init_stdio().unwrap();
    let result = lx.handle_syscall(0, [0, 0, 0, 0, 0, 0]);
    assert_eq!(result, 0);
}

#[test]
fn sys_close_removes_fd() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    lx.init_stdio().unwrap();
    assert!(lx.has_fd(1));

    let result = lx.handle_syscall(3, [1, 0, 0, 0, 0, 0]); // close(1)
    assert_eq!(result, 0);
    assert!(!lx.has_fd(1));

    // Backend should have received clunk
    assert_eq!(lx.backend().clunks.len(), 1);
}

#[test]
fn sys_close_bad_fd() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    let result = lx.handle_syscall(3, [99, 0, 0, 0, 0, 0]);
    assert_eq!(result, EBADF);
}

#[test]
fn sys_openat_walks_and_opens() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    lx.init_stdio().unwrap();

    let path = b"/dev/serial/log\0";
    // openat(AT_FDCWD, path, O_RDONLY)
    let at_fdcwd = (-100i32) as u64;
    let result = lx.handle_syscall(257, [at_fdcwd, path.as_ptr() as u64, 0, 0, 0, 0]);
    assert!(result >= 0);
    assert_eq!(result, 3); // fd 0,1,2 taken → next is 3

    // Verify walk and open happened
    assert!(lx.backend().walks.len() > 3); // 3 from init_stdio + 1 from openat
    assert!(lx.has_fd(3));
}

#[test]
fn sys_exit_is_same_as_exit_group() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    let result = lx.handle_syscall(60, [7, 0, 0, 0, 0, 0]); // exit(7)
    assert_eq!(result, 0);
    assert!(lx.exited());
    assert_eq!(lx.exit_code(), Some(7));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-os sys_read sys_close sys_openat sys_exit_is`
Expected: FAIL — methods don't exist

**Step 3: Implement**

1. Add helper to read null-terminated string from process memory:
```rust
/// Read a null-terminated C string from a pointer in the flat address space.
///
/// # Safety
/// `ptr` must point to a valid null-terminated string.
unsafe fn read_c_string(ptr: usize) -> &'static str {
    let p = ptr as *const u8;
    let mut len = 0;
    while *p.add(len) != 0 {
        len += 1;
    }
    core::str::from_utf8_unchecked(core::slice::from_raw_parts(p, len))
}
```

2. Add `alloc_fd` helper:
```rust
fn alloc_fd(&self) -> i32 {
    let mut fd = 0;
    while self.fd_table.contains_key(&fd) {
        fd += 1;
    }
    fd
}
```

3. Add `flags_to_open_mode` helper (as a free function, before the Linuxulator impl block):
```rust
fn flags_to_open_mode(flags: i32) -> OpenMode {
    let accmode = flags & 0x03; // O_ACCMODE
    match accmode {
        0 => OpenMode::Read,      // O_RDONLY
        1 => OpenMode::Write,     // O_WRONLY
        2 => OpenMode::ReadWrite, // O_RDWR
        _ => OpenMode::Read,
    }
}
```

4. Implement sys_read, sys_close, sys_openat, sys_exit:
```rust
fn sys_read(&mut self, fd: i32, buf_ptr: usize, count: usize) -> i64 {
    if count == 0 {
        return 0;
    }
    let fid = match self.fd_table.get(&fd) {
        Some(&fid) => fid,
        None => return EBADF,
    };
    match self.backend.read(fid, 0, count as u32) {
        Ok(data) => {
            let n = data.len();
            if n > 0 {
                unsafe {
                    core::ptr::copy_nonoverlapping(data.as_ptr(), buf_ptr as *mut u8, n);
                }
            }
            n as i64
        }
        Err(e) => ipc_err_to_errno(e),
    }
}

fn sys_close(&mut self, fd: i32) -> i64 {
    let fid = match self.fd_table.remove(&fd) {
        Some(f) => f,
        None => return EBADF,
    };
    let _ = self.backend.clunk(fid);
    0
}

fn sys_openat(&mut self, dirfd: i32, pathname_ptr: usize, flags: i32) -> i64 {
    let path = unsafe { read_c_string(pathname_ptr) };
    let at_fdcwd: i32 = -100;
    if dirfd != at_fdcwd && !self.fd_table.contains_key(&dirfd) {
        return EBADF;
    }
    let fid = self.alloc_fid();
    let mode = flags_to_open_mode(flags);
    if let Err(e) = self.backend.walk(path, fid) {
        return ipc_err_to_errno(e);
    }
    if let Err(e) = self.backend.open(fid, mode) {
        let _ = self.backend.clunk(fid);
        return ipc_err_to_errno(e);
    }
    let fd = self.alloc_fd();
    self.fd_table.insert(fd, fid);
    fd as i64
}

fn sys_exit(&mut self, code: i32) -> i64 {
    self.sys_exit_group(code)
}
```

5. Wire into `handle_syscall`:
```rust
match nr {
    0 => self.sys_read(args[0] as i32, args[1] as usize, args[2] as usize),
    1 => self.sys_write(args[0] as i32, args[1] as usize, args[2] as usize),
    3 => self.sys_close(args[0] as i32),
    9 => self.sys_mmap(args[0], args[1], args[2] as i32, args[3] as i32, args[4] as i32, args[5]),
    11 => self.sys_munmap(args[0], args[1]),
    12 => self.sys_brk(args[0]),
    60 => self.sys_exit(args[0] as i32),
    231 => self.sys_exit_group(args[0] as i32),
    257 => self.sys_openat(args[0] as i32, args[1] as usize, args[2] as i32),
    _ => ENOSYS,
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-os`
Expected: ALL tests pass

**Step 5: Commit**

```bash
git add crates/harmony-os/src/linuxulator.rs
git commit -m "feat(linuxulator): sys_read, sys_close, sys_openat, sys_exit"
```

---

### Task 4: fstat — write Linux struct stat to process memory

**Files:**
- Modify: `crates/harmony-os/src/linuxulator.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn sys_fstat_writes_stat_struct() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    lx.init_stdio().unwrap();

    let mut statbuf = [0u8; 144]; // Linux struct stat is 144 bytes
    let result = lx.handle_syscall(5, [1, statbuf.as_mut_ptr() as u64, 0, 0, 0, 0]);
    assert_eq!(result, 0);

    // st_mode should be S_IFCHR | 0o666 for stdio fds
    let st_mode = u32::from_le_bytes([statbuf[24], statbuf[25], statbuf[26], statbuf[27]]);
    let s_ifchr: u32 = 0o020000;
    assert_eq!(st_mode & 0o170000, s_ifchr); // file type is char device
}

#[test]
fn sys_fstat_bad_fd() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    let mut statbuf = [0u8; 144];
    let result = lx.handle_syscall(5, [99, statbuf.as_mut_ptr() as u64, 0, 0, 0, 0]);
    assert_eq!(result, EBADF);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-os sys_fstat`
Expected: FAIL

**Step 3: Implement**

1. Add `write_linux_stat` helper and `sys_fstat`. The key insight: stdio fds (0, 1, 2) should report `S_IFCHR` (character device), other fds use the `FileType` from `FileStat`:

```rust
/// Linux struct stat field offsets (x86_64, 144 bytes total).
fn write_linux_stat(buf_ptr: usize, stat: &FileStat, is_chardev: bool) {
    let buf = unsafe { core::slice::from_raw_parts_mut(buf_ptr as *mut u8, 144) };
    // Zero everything first
    buf.fill(0);

    // st_ino (offset 8, 8 bytes)
    buf[8..16].copy_from_slice(&stat.qpath.to_le_bytes());

    // st_nlink (offset 16, 8 bytes)
    buf[16..24].copy_from_slice(&1u64.to_le_bytes());

    // st_mode (offset 24, 4 bytes)
    let mode: u32 = if is_chardev {
        0o020000 | 0o666 // S_IFCHR | rw-rw-rw-
    } else {
        match stat.file_type {
            FileType::Regular => 0o100000 | 0o644,   // S_IFREG | rw-r--r--
            FileType::Directory => 0o040000 | 0o755,  // S_IFDIR | rwxr-xr-x
        }
    };
    buf[24..28].copy_from_slice(&mode.to_le_bytes());

    // st_size (offset 48, 8 bytes)
    buf[48..56].copy_from_slice(&stat.size.to_le_bytes());

    // st_blksize (offset 56, 8 bytes)
    buf[56..64].copy_from_slice(&4096u64.to_le_bytes());

    // st_blocks (offset 64, 8 bytes)
    let blocks = (stat.size + 511) / 512;
    buf[64..72].copy_from_slice(&blocks.to_le_bytes());
}

fn sys_fstat(&mut self, fd: i32, statbuf_ptr: usize) -> i64 {
    let fid = match self.fd_table.get(&fd) {
        Some(&fid) => fid,
        None => return EBADF,
    };
    let is_chardev = fd <= 2; // stdin/stdout/stderr are character devices
    match self.backend.stat(fid) {
        Ok(stat) => {
            write_linux_stat(statbuf_ptr, &stat, is_chardev);
            0
        }
        Err(e) => ipc_err_to_errno(e),
    }
}
```

2. Add `stat` to the SyscallBackend implementation for `KernelBackend` in integration_tests:
```rust
fn stat(&mut self, fid: Fid) -> Result<FileStat, IpcError> {
    self.kernel.stat(self.pid, fid)
}
```

3. Wire `fstat` into `handle_syscall`:
```rust
5 => self.sys_fstat(args[0] as i32, args[1] as usize),
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-os`
Expected: ALL tests pass

**Step 5: Commit**

```bash
git add crates/harmony-os/src/linuxulator.rs
git commit -m "feat(linuxulator): sys_fstat with Linux struct stat layout"
```

---

### Task 5: Stub syscalls — ioctl, sigaction, sigprocmask, set_tid_address, set_robust_list, prlimit64, rseq

**Files:**
- Modify: `crates/harmony-os/src/linuxulator.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn sys_ioctl_tiocgwinsz_returns_enotty() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    lx.init_stdio().unwrap();
    let result = lx.handle_syscall(16, [1, 0x5413, 0, 0, 0, 0]); // ioctl(stdout, TIOCGWINSZ)
    assert_eq!(result, ENOTTY);
}

#[test]
fn sys_ioctl_unknown_returns_enosys() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    lx.init_stdio().unwrap();
    let result = lx.handle_syscall(16, [1, 0xFFFF, 0, 0, 0, 0]);
    assert_eq!(result, ENOSYS);
}

#[test]
fn sys_set_tid_address_returns_tid() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    let result = lx.handle_syscall(218, [0, 0, 0, 0, 0, 0]);
    assert_eq!(result, 1); // TID = 1
}

#[test]
fn sys_set_robust_list_returns_zero() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    let result = lx.handle_syscall(273, [0, 0, 0, 0, 0, 0]);
    assert_eq!(result, 0);
}

#[test]
fn sys_rt_sigaction_returns_zero() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    let result = lx.handle_syscall(13, [2, 0, 0, 8, 0, 0]); // sigaction(SIGINT, ...)
    assert_eq!(result, 0);
}

#[test]
fn sys_rt_sigprocmask_returns_zero() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    let result = lx.handle_syscall(14, [0, 0, 0, 0, 0, 0]);
    assert_eq!(result, 0);
}

#[test]
fn sys_prlimit64_writes_stack_limit() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    let mut rlimit = [0u8; 16]; // rlim_cur (8) + rlim_max (8)
    // prlimit64(0, RLIMIT_STACK=3, NULL, &rlimit)
    let result = lx.handle_syscall(302, [0, 3, 0, rlimit.as_mut_ptr() as u64, 0, 0]);
    assert_eq!(result, 0);
    let rlim_cur = u64::from_le_bytes(rlimit[0..8].try_into().unwrap());
    assert_eq!(rlim_cur, 8 * 1024 * 1024); // 8 MiB
}

#[test]
fn sys_rseq_returns_enosys() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    let result = lx.handle_syscall(334, [0, 0, 0, 0, 0, 0]);
    assert_eq!(result, ENOSYS);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-os sys_ioctl sys_set_tid sys_set_robust sys_rt_sig sys_prlimit sys_rseq`
Expected: FAIL

**Step 3: Implement**

Add stub methods and wire into handle_syscall:

```rust
fn sys_ioctl(&self, _fd: i32, request: u64) -> i64 {
    const TIOCGWINSZ: u64 = 0x5413;
    match request {
        TIOCGWINSZ => ENOTTY,
        _ => ENOSYS,
    }
}

fn sys_rt_sigaction(&self) -> i64 { 0 }
fn sys_rt_sigprocmask(&self) -> i64 { 0 }
fn sys_set_tid_address(&self) -> i64 { 1 } // TID = 1
fn sys_set_robust_list(&self) -> i64 { 0 }

fn sys_prlimit64(&self, pid: i32, resource: i32, _new_limit: u64, old_limit_ptr: usize) -> i64 {
    const RLIMIT_STACK: i32 = 3;
    if pid != 0 { return ESRCH; }
    if resource == RLIMIT_STACK && old_limit_ptr != 0 {
        let eight_mb = 8u64 * 1024 * 1024;
        unsafe {
            *(old_limit_ptr as *mut u64) = eight_mb;       // rlim_cur
            *((old_limit_ptr + 8) as *mut u64) = eight_mb; // rlim_max
        }
    }
    0
}
```

Update `handle_syscall`:
```rust
match nr {
    0   => self.sys_read(args[0] as i32, args[1] as usize, args[2] as usize),
    1   => self.sys_write(args[0] as i32, args[1] as usize, args[2] as usize),
    3   => self.sys_close(args[0] as i32),
    5   => self.sys_fstat(args[0] as i32, args[1] as usize),
    9   => self.sys_mmap(args[0], args[1], args[2] as i32, args[3] as i32, args[4] as i32, args[5]),
    11  => self.sys_munmap(args[0], args[1]),
    12  => self.sys_brk(args[0]),
    13  => self.sys_rt_sigaction(),
    14  => self.sys_rt_sigprocmask(),
    16  => self.sys_ioctl(args[0] as i32, args[1]),
    60  => self.sys_exit(args[0] as i32),
    158 => ENOSYS, // arch_prctl — placeholder, implemented in Task 6
    218 => self.sys_set_tid_address(),
    231 => self.sys_exit_group(args[0] as i32),
    257 => self.sys_openat(args[0] as i32, args[1] as usize, args[2] as i32),
    273 => self.sys_set_robust_list(),
    302 => self.sys_prlimit64(args[0] as i32, args[1] as i32, args[2], args[3] as usize),
    334 => ENOSYS, // rseq — musl handles gracefully
    _   => ENOSYS,
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-os`
Expected: ALL tests pass

**Step 5: Commit**

```bash
git add crates/harmony-os/src/linuxulator.rs
git commit -m "feat(linuxulator): stub syscalls — ioctl, sigaction, tid, robust_list, prlimit64"
```

---

### Task 6: arch_prctl — FS base register for TLS

**Files:**
- Modify: `crates/harmony-boot/src/syscall.rs` (add write_fs_base/read_fs_base)
- Modify: `crates/harmony-os/src/linuxulator.rs` (add sys_arch_prctl with platform abstraction)

**Step 1: Write failing test**

In `linuxulator.rs` tests:

```rust
#[test]
fn sys_arch_prctl_set_fs_returns_zero() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    // ARCH_SET_FS = 0x1002
    let result = lx.handle_syscall(158, [0x1002, 0x12345678, 0, 0, 0, 0]);
    assert_eq!(result, 0);
}

#[test]
fn sys_arch_prctl_unknown_code_returns_einval() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    let result = lx.handle_syscall(158, [0x9999, 0, 0, 0, 0, 0]);
    assert_eq!(result, EINVAL);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-os sys_arch_prctl`
Expected: FAIL

**Step 3: Implement**

1. In `linuxulator.rs`, add arch_prctl. On the test host (non-bare-metal), the FS base write is a no-op — we can't write to MSRs from user space. The actual `wrfsbase`/`wrmsr` happens only on bare metal via `harmony-boot::syscall`:

```rust
fn sys_arch_prctl(&mut self, code: i32, addr: u64) -> i64 {
    const ARCH_SET_FS: i32 = 0x1002;
    const ARCH_GET_FS: i32 = 0x1003;
    match code {
        ARCH_SET_FS => {
            self.fs_base = addr;
            // On bare metal, the boot crate writes the actual MSR.
            // In unit tests, we just record the value.
            0
        }
        ARCH_GET_FS => {
            if addr != 0 {
                unsafe { *(addr as *mut u64) = self.fs_base; }
            }
            0
        }
        _ => EINVAL,
    }
}
```

2. Add `fs_base: u64` field to `Linuxulator` struct, initialize to 0 in `new`/`with_arena`.

3. Add public accessor for boot integration:
```rust
pub fn fs_base(&self) -> u64 {
    self.fs_base
}
```

4. Wire into handle_syscall:
```rust
158 => self.sys_arch_prctl(args[0] as i32, args[1]),
```

5. In `crates/harmony-boot/src/syscall.rs`, add FS base functions after the MSR setup section:

```rust
// ── FS base register ──────────────────────────────────────────────

const IA32_FS_BASE: u32 = 0xC000_0100;

/// Write the FS segment base register (for TLS).
///
/// # Safety
/// Must be called in Ring 0.
pub unsafe fn write_fs_base(addr: u64) {
    wrmsr(IA32_FS_BASE, addr);
}

/// Read the FS segment base register.
///
/// # Safety
/// Must be called in Ring 0.
pub unsafe fn read_fs_base() -> u64 {
    rdmsr(IA32_FS_BASE)
}
```

6. Update the `dispatch` function in `main.rs` to write the FS base after each syscall that might set it. Add to the dispatch function body, after `lx.handle_syscall`:

```rust
fn dispatch(nr: u64, args: [u64; 6]) -> syscall::SyscallResult {
    let lx = unsafe { LINUXULATOR.as_mut().unwrap() };
    // For arch_prctl(ARCH_SET_FS), write the actual MSR
    if nr == 158 && args[0] == 0x1002 {
        unsafe { syscall::write_fs_base(args[1]); }
    }
    let retval = lx.handle_syscall(nr, args);
    syscall::SyscallResult {
        retval,
        exited: lx.exited(),
        exit_code: lx.exit_code().unwrap_or(0),
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-os`
Expected: ALL tests pass

**Step 5: Commit**

```bash
git add crates/harmony-os/src/linuxulator.rs crates/harmony-boot/src/syscall.rs crates/harmony-boot/src/main.rs
git commit -m "feat(linuxulator): arch_prctl for TLS, FS base MSR support"
```

---

### Task 7: Update DirectBackend for stat + integration test expansion

**Files:**
- Modify: `crates/harmony-boot/src/main.rs` (add stat to DirectBackend)
- Modify: `crates/harmony-os/src/linuxulator.rs` (expand integration test)

**Step 1: Write failing integration test**

Expand the integration test in `linuxulator.rs`:

```rust
#[test]
fn linuxulator_full_fd_lifecycle() {
    let mut entropy = test_entropy();
    let kernel_id = PrivateIdentity::generate(&mut entropy);
    let mut kernel = Kernel::new(kernel_id);

    let serial_pid = kernel
        .spawn_process("serial", Box::new(SerialServer::new()), &[])
        .unwrap();

    let linux_pid = kernel
        .spawn_process(
            "hello-linux",
            Box::new(EchoServer::new()),
            &[("/dev/serial", serial_pid, 0)],
        )
        .unwrap();

    kernel
        .grant_endpoint_cap(&mut entropy, linux_pid, serial_pid, 0)
        .unwrap();

    let backend = KernelBackend::new(&mut kernel, linux_pid);
    let mut lx = Linuxulator::new(backend);
    lx.init_stdio().unwrap();

    // Write to stdout
    let msg = b"Hello\n";
    let result = lx.handle_syscall(1, [1, msg.as_ptr() as u64, 6, 0, 0, 0]);
    assert_eq!(result, 6);

    // fstat on stdout — should succeed
    let mut statbuf = [0u8; 144];
    let result = lx.handle_syscall(5, [1, statbuf.as_mut_ptr() as u64, 0, 0, 0, 0]);
    assert_eq!(result, 0);

    // Close stdout
    let result = lx.handle_syscall(3, [1, 0, 0, 0, 0, 0]);
    assert_eq!(result, 0);
    assert!(!lx.has_fd(1));

    // Write to closed fd should fail
    let result = lx.handle_syscall(1, [1, msg.as_ptr() as u64, 6, 0, 0, 0]);
    assert_eq!(result, -9); // EBADF

    // Exit
    lx.handle_syscall(231, [0, 0, 0, 0, 0, 0]);
    assert!(lx.exited());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-os linuxulator_full_fd_lifecycle`
Expected: FAIL — `KernelBackend` doesn't have `stat`

**Step 3: Implement**

1. Add `stat` to `KernelBackend` in the integration_tests module (it was added to the trait in Task 1 but not implemented here):
```rust
fn stat(&mut self, fid: Fid) -> Result<FileStat, IpcError> {
    self.kernel.stat(self.pid, fid)
}
```

2. Add `stat` to `DirectBackend` in `main.rs`:
```rust
fn stat(
    &mut self,
    fid: harmony_microkernel::Fid,
) -> Result<harmony_microkernel::FileStat, harmony_microkernel::IpcError> {
    self.server.stat(fid)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-os`
Expected: ALL tests pass

**Step 5: Commit**

```bash
git add crates/harmony-os/src/linuxulator.rs crates/harmony-boot/src/main.rs
git commit -m "feat(linuxulator): stat on backends, full fd lifecycle integration test"
```

---

### Task 8: Update sys_write to use ipc_err_to_errno

**Files:**
- Modify: `crates/harmony-os/src/linuxulator.rs`

This is a small cleanup — the existing `sys_write` returns `EIO` on any error. Now that we have `ipc_err_to_errno`, use it for consistency.

**Step 1: Verify existing test still passes after change**

No new test needed — existing `sys_write_bad_fd` and `sys_write_to_stdout` cover this.

**Step 2: Implement**

Change `sys_write` error handling from:
```rust
Err(_) => EIO,
```
to:
```rust
Err(e) => ipc_err_to_errno(e),
```

**Step 3: Run tests to verify they pass**

Run: `cargo test -p harmony-os`
Expected: ALL tests pass

**Step 4: Commit**

```bash
git add crates/harmony-os/src/linuxulator.rs
git commit -m "refactor(linuxulator): sys_write uses ipc_err_to_errno for consistent error mapping"
```

---

### Task 9: Quality gates

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `cargo test --workspace`
Expected: ALL tests pass (154+ existing + ~20 new)

**Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: Zero warnings

**Step 3: Run fmt check**

Run: `cargo fmt --all -- --check`
Expected: No formatting issues

**Step 4: Fix any issues found**

If any warnings or test failures, fix them before proceeding.

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore: quality gate fixes"
```

(Only if there are fixes needed. Skip this commit if everything passes clean.)

---

## Summary

| Task | What | Lines (est.) |
|------|------|-------------|
| 1 | stat on SyscallBackend, ipc_err_to_errno, errno constants | +60 |
| 2 | MemoryArena, brk, mmap, munmap | +150 |
| 3 | sys_read, sys_close, sys_openat, sys_exit, helpers | +100 |
| 4 | sys_fstat, write_linux_stat | +70 |
| 5 | Stub syscalls (ioctl, sigaction, etc.) | +60 |
| 6 | arch_prctl, FS base MSR, dispatch update | +50 |
| 7 | DirectBackend stat, integration test | +40 |
| 8 | sys_write error handling cleanup | +2 |
| 9 | Quality gates | 0 |
| **Total** | | **~530 new lines** |

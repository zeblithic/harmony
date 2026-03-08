# Linuxulator MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Get a statically-linked x86_64 Linux binary to print "Hello\n" via 9P SerialServer on the QEMU serial console, with Linux syscalls translated through the Linuxulator.

**Architecture:** ELF parser and Linuxulator live in `harmony-os` as testable `no_std` library code. The x86_64 syscall trap and QEMU glue live in `harmony-boot`. The Linuxulator translates Linux syscalls to 9P operations via a `SyscallBackend` trait — mocked for unit tests, wired to a real SerialServer for integration.

**Tech Stack:** Rust, `no_std + alloc`, x86_64 inline assembly (MSR setup, naked syscall handler), GNU `as`/`ld` for the test binary.

**Design doc:** `docs/plans/2026-03-08-linuxulator-design.md`

**Repos involved:** `harmony-os` only (implementation). `harmony` (this repo) holds only the design/plan docs.

---

### Task 1: Add `no_std` feature gating to harmony-os

**Files:**
- Modify: `crates/harmony-os/Cargo.toml`
- Modify: `crates/harmony-os/src/lib.rs`

**Context:** The ELF parser and Linuxulator modules use only `alloc` types and harmony-microkernel's base types (Fid, IpcError, OpenMode). These are all `no_std`-compatible. But harmony-os currently hard-codes `features = ["std"]` on all Ring 0 deps. We need to gate those behind a `std` feature so harmony-boot can depend on harmony-os with `default-features = false`.

**Step 1: Update Cargo.toml**

Replace the current `[dependencies]` with feature-gated deps:

```toml
[package]
name = "harmony-os"
description = "Ring 3: Full Harmony OS with Linux ABI compatibility, DDE, declarative config"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[features]
default = ["std"]
std = [
    "harmony-microkernel/kernel",
    "dep:harmony-crypto", "harmony-crypto/std",
    "dep:harmony-identity", "harmony-identity/std",
    "dep:harmony-reticulum", "harmony-reticulum/std",
    "dep:harmony-zenoh",
    "dep:harmony-content",
    "dep:harmony-compute",
    "dep:harmony-workflow",
]

[dependencies]
# Ring 2 — base types (Fid, FileServer, IpcError) are always available
harmony-microkernel = { workspace = true, default-features = false }

# Ring 0 — full protocol stack, only with std
harmony-crypto = { workspace = true, optional = true }
harmony-identity = { workspace = true, optional = true }
harmony-reticulum = { workspace = true, optional = true }
harmony-zenoh = { workspace = true, optional = true }
harmony-content = { workspace = true, optional = true }
harmony-compute = { workspace = true, optional = true }
harmony-workflow = { workspace = true, optional = true }

[dev-dependencies]
```

**Step 2: Update lib.rs**

```rust
// SPDX-License-Identifier: GPL-2.0-or-later

//! # Harmony OS (Ring 3)
//!
//! Full operating system built on the microkernel foundation.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(test)]
mod tests {
    #[test]
    fn ring3_placeholder() {
        assert!(true);
    }
}
```

**Step 3: Verify it compiles both ways**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo check -p harmony-os`
Expected: compiles (with default std feature)

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo check -p harmony-os --no-default-features`
Expected: compiles (no_std, just alloc + microkernel types)

**Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-os/Cargo.toml crates/harmony-os/src/lib.rs
git commit -m "feat(os): add no_std feature gating for Ring 3"
```

---

### Task 2: ELF parser

**Files:**
- Create: `crates/harmony-os/src/elf.rs`
- Modify: `crates/harmony-os/src/lib.rs` (add `pub mod elf;`)

**Context:** Parse statically-linked x86_64 ELF64 executables. Only supports ET_EXEC (not ET_DYN). No dynamic linking. No section headers needed — only program headers (PT_LOAD). The parser reads raw bytes, validates the header, extracts segments. Memory allocation (for loading segments) is the caller's responsibility — the parser returns metadata only.

A valid ELF64 header is 64 bytes. Each program header is 56 bytes. We parse manually (no external crate) to stay `no_std`.

**Step 1: Write the failing tests**

Add tests to the bottom of `elf.rs` (they reference types that don't exist yet):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid ELF64 binary in memory for testing.
    /// `code` is placed as a PT_LOAD segment at vaddr 0x401000.
    fn build_test_elf(code: &[u8]) -> Vec<u8> {
        let mut elf = vec![0u8; 64 + 56 + code.len()];

        // ELF magic
        elf[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        elf[4] = 2;    // ELFCLASS64
        elf[5] = 1;    // ELFDATA2LSB
        elf[6] = 1;    // EV_CURRENT
        // e_type = ET_EXEC (2)
        elf[16..18].copy_from_slice(&2u16.to_le_bytes());
        // e_machine = EM_X86_64 (0x3E)
        elf[18..20].copy_from_slice(&0x3Eu16.to_le_bytes());
        // e_version
        elf[20..24].copy_from_slice(&1u32.to_le_bytes());
        // e_entry = 0x401000
        elf[24..32].copy_from_slice(&0x401000u64.to_le_bytes());
        // e_phoff = 64 (right after header)
        elf[32..40].copy_from_slice(&64u64.to_le_bytes());
        // e_shoff = 0 (no section headers)
        // e_flags = 0
        // e_ehsize = 64
        elf[52..54].copy_from_slice(&64u16.to_le_bytes());
        // e_phentsize = 56
        elf[54..56].copy_from_slice(&56u16.to_le_bytes());
        // e_phnum = 1
        elf[56..58].copy_from_slice(&1u16.to_le_bytes());
        // e_shentsize = 0, e_shnum = 0, e_shstrndx = 0

        // Program header (PT_LOAD)
        let ph = &mut elf[64..120];
        // p_type = PT_LOAD (1)
        ph[0..4].copy_from_slice(&1u32.to_le_bytes());
        // p_flags = PF_R | PF_X (5)
        ph[4..8].copy_from_slice(&5u32.to_le_bytes());
        // p_offset = 120 (after headers)
        ph[8..16].copy_from_slice(&120u64.to_le_bytes());
        // p_vaddr = 0x401000
        ph[16..24].copy_from_slice(&0x401000u64.to_le_bytes());
        // p_paddr = 0x401000
        ph[24..32].copy_from_slice(&0x401000u64.to_le_bytes());
        // p_filesz = code.len()
        ph[32..40].copy_from_slice(&(code.len() as u64).to_le_bytes());
        // p_memsz = code.len()
        ph[40..48].copy_from_slice(&(code.len() as u64).to_le_bytes());
        // p_align = 0x1000
        ph[48..56].copy_from_slice(&0x1000u64.to_le_bytes());

        // Code
        elf[120..120 + code.len()].copy_from_slice(code);

        elf
    }

    #[test]
    fn parse_valid_elf() {
        let code = [0xCC; 16]; // dummy code
        let elf = build_test_elf(&code);
        let parsed = parse_elf(&elf).unwrap();
        assert_eq!(parsed.entry_point, 0x401000);
        assert_eq!(parsed.segments.len(), 1);
        assert_eq!(parsed.segments[0].vaddr, 0x401000);
        assert_eq!(parsed.segments[0].filesz, 16);
        assert_eq!(parsed.segments[0].memsz, 16);
        assert!(parsed.segments[0].flags.execute);
        assert!(parsed.segments[0].flags.read);
        assert!(!parsed.segments[0].flags.write);
    }

    #[test]
    fn segment_data_correct() {
        let code = [0x90, 0x90, 0xCC]; // nop, nop, int3
        let elf = build_test_elf(&code);
        let parsed = parse_elf(&elf).unwrap();
        let seg = &parsed.segments[0];
        let data = &elf[seg.offset as usize..(seg.offset + seg.filesz) as usize];
        assert_eq!(data, &[0x90, 0x90, 0xCC]);
    }

    #[test]
    fn reject_bad_magic() {
        let mut elf = build_test_elf(&[0xCC]);
        elf[0] = 0x00; // corrupt magic
        assert_eq!(parse_elf(&elf), Err(ElfError::BadMagic));
    }

    #[test]
    fn reject_32bit_elf() {
        let mut elf = build_test_elf(&[0xCC]);
        elf[4] = 1; // ELFCLASS32
        assert_eq!(parse_elf(&elf), Err(ElfError::Not64Bit));
    }

    #[test]
    fn reject_non_x86_64() {
        let mut elf = build_test_elf(&[0xCC]);
        elf[18..20].copy_from_slice(&0x03u16.to_le_bytes()); // EM_386
        assert_eq!(parse_elf(&elf), Err(ElfError::NotX86_64));
    }

    #[test]
    fn reject_dynamic_elf() {
        let mut elf = build_test_elf(&[0xCC]);
        elf[16..18].copy_from_slice(&3u16.to_le_bytes()); // ET_DYN
        assert_eq!(parse_elf(&elf), Err(ElfError::NotExecutable));
    }

    #[test]
    fn reject_truncated_header() {
        let elf = vec![0x7f, b'E', b'L', b'F']; // too short
        assert_eq!(parse_elf(&elf), Err(ElfError::TooShort));
    }

    #[test]
    fn bss_segment_has_memsz_greater_than_filesz() {
        let code = [0xCC; 16];
        let mut elf = build_test_elf(&code);
        // Set memsz > filesz (simulates .bss)
        let ph_memsz = &mut elf[64 + 40..64 + 48];
        ph_memsz.copy_from_slice(&256u64.to_le_bytes());
        let parsed = parse_elf(&elf).unwrap();
        assert_eq!(parsed.segments[0].filesz, 16);
        assert_eq!(parsed.segments[0].memsz, 256);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os elf::tests`
Expected: FAIL — module doesn't exist

**Step 3: Write the implementation**

Create `crates/harmony-os/src/elf.rs`:

```rust
//! Minimal ELF64 parser for statically-linked x86_64 executables.
//!
//! Only supports ET_EXEC binaries with PT_LOAD segments. No dynamic
//! linking, no section headers, no interpreter. Returns metadata
//! only — the caller allocates memory and copies segments.

use alloc::vec::Vec;

// ── ELF constants ───────────────────────────────────────────────────

const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const ET_EXEC: u16 = 2;
const EM_X86_64: u16 = 0x3E;
const PT_LOAD: u32 = 1;

const ELF64_HEADER_SIZE: usize = 64;
const ELF64_PHDR_SIZE: usize = 56;

// ── Segment permission flags ────────────────────────────────────────

const PF_X: u32 = 1;
const PF_W: u32 = 2;
const PF_R: u32 = 4;

// ── Error type ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElfError {
    TooShort,
    BadMagic,
    Not64Bit,
    NotLittleEndian,
    NotExecutable,
    NotX86_64,
    InvalidPhdr,
    SegmentOutOfBounds,
}

// ── Parsed types ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct SegmentFlags {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

#[derive(Debug, Clone)]
pub struct ElfSegment {
    /// Virtual address the segment expects to be loaded at.
    pub vaddr: u64,
    /// Offset into the ELF file where segment data starts.
    pub offset: u64,
    /// Bytes to copy from the ELF file.
    pub filesz: u64,
    /// Total memory size (filesz + zero-fill for .bss).
    pub memsz: u64,
    /// Segment permissions.
    pub flags: SegmentFlags,
    /// Requested alignment.
    pub align: u64,
}

/// Parsed ELF metadata. Does not contain the actual segment data —
/// use `offset` and `filesz` to slice the original ELF bytes.
#[derive(Debug)]
pub struct ParsedElf {
    pub entry_point: u64,
    pub segments: Vec<ElfSegment>,
}

// ── Little-endian helpers ───────────────────────────────────────────

fn u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn u64_le(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

// ── Parser ──────────────────────────────────────────────────────────

/// Parse an ELF64 binary from raw bytes.
///
/// Returns metadata about loadable segments and the entry point.
/// Only supports statically-linked x86_64 ET_EXEC binaries.
pub fn parse_elf(data: &[u8]) -> Result<ParsedElf, ElfError> {
    if data.len() < ELF64_HEADER_SIZE {
        return Err(ElfError::TooShort);
    }

    // Validate ELF magic
    if data[0..4] != ELF_MAGIC {
        return Err(ElfError::BadMagic);
    }

    // Must be 64-bit
    if data[4] != ELFCLASS64 {
        return Err(ElfError::Not64Bit);
    }

    // Must be little-endian
    if data[5] != ELFDATA2LSB {
        return Err(ElfError::NotLittleEndian);
    }

    // Must be ET_EXEC
    let e_type = u16_le(data, 16);
    if e_type != ET_EXEC {
        return Err(ElfError::NotExecutable);
    }

    // Must be x86_64
    let e_machine = u16_le(data, 18);
    if e_machine != EM_X86_64 {
        return Err(ElfError::NotX86_64);
    }

    let entry_point = u64_le(data, 24);
    let e_phoff = u64_le(data, 32) as usize;
    let e_phentsize = u16_le(data, 54) as usize;
    let e_phnum = u16_le(data, 56) as usize;

    // Validate program header table fits
    if e_phentsize < ELF64_PHDR_SIZE {
        return Err(ElfError::InvalidPhdr);
    }
    let ph_end = e_phoff
        .checked_add(e_phnum.checked_mul(e_phentsize).ok_or(ElfError::InvalidPhdr)?)
        .ok_or(ElfError::InvalidPhdr)?;
    if ph_end > data.len() {
        return Err(ElfError::TooShort);
    }

    // Parse PT_LOAD segments
    let mut segments = Vec::new();
    for i in 0..e_phnum {
        let ph = &data[e_phoff + i * e_phentsize..];
        let p_type = u32_le(ph, 0);
        if p_type != PT_LOAD {
            continue;
        }

        let p_flags = u32_le(ph, 4);
        let p_offset = u64_le(ph, 8);
        let p_vaddr = u64_le(ph, 16);
        let p_filesz = u64_le(ph, 32);
        let p_memsz = u64_le(ph, 40);
        let p_align = u64_le(ph, 48);

        // Validate segment data fits in the ELF
        let seg_end = p_offset
            .checked_add(p_filesz)
            .ok_or(ElfError::SegmentOutOfBounds)?;
        if seg_end as usize > data.len() {
            return Err(ElfError::SegmentOutOfBounds);
        }

        segments.push(ElfSegment {
            vaddr: p_vaddr,
            offset: p_offset,
            filesz: p_filesz,
            memsz: p_memsz,
            flags: SegmentFlags {
                read: p_flags & PF_R != 0,
                write: p_flags & PF_W != 0,
                execute: p_flags & PF_X != 0,
            },
            align: p_align,
        });
    }

    Ok(ParsedElf {
        entry_point,
        segments,
    })
}
```

Add to `crates/harmony-os/src/lib.rs`:

```rust
pub mod elf;
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os elf::tests`
Expected: all 8 tests PASS

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-os/src/elf.rs crates/harmony-os/src/lib.rs
git commit -m "feat(os): add ELF64 parser for statically-linked x86_64 binaries"
```

---

### Task 3: SyscallBackend trait and MockBackend

**Files:**
- Create: `crates/harmony-os/src/linuxulator.rs`
- Modify: `crates/harmony-os/src/lib.rs` (add `pub mod linuxulator;`)

**Context:** The `SyscallBackend` trait abstracts 9P operations so the Linuxulator is testable without a real Kernel. The trait uses types from harmony-microkernel (Fid, OpenMode, IpcError) which are available even without `std`. The `MockBackend` records all calls for test assertions.

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn mock_backend_records_write() {
        let mut mock = MockBackend::new();
        mock.write(1, 0, b"hello").unwrap();
        assert_eq!(mock.writes.len(), 1);
        assert_eq!(mock.writes[0], (1, vec![b'h', b'e', b'l', b'l', b'o']));
    }

    #[test]
    fn mock_backend_records_walk() {
        let mut mock = MockBackend::new();
        let qpath = mock.walk("/dev/serial/log", 10).unwrap();
        assert_eq!(qpath, 0);
        assert_eq!(mock.walks.len(), 1);
        assert_eq!(mock.walks[0], ("/dev/serial/log".into(), 10));
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os linuxulator::tests::mock_`
Expected: FAIL — module doesn't exist

**Step 3: Write the implementation**

Create `crates/harmony-os/src/linuxulator.rs`:

```rust
//! Linuxulator — Linux syscall-to-9P translation layer for Ring 3.
//!
//! Translates Linux syscall numbers and arguments into 9P FileServer
//! operations via a [`SyscallBackend`] trait. Manages a POSIX-style
//! fd table that maps Linux file descriptors to 9P fids.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use harmony_microkernel::{Fid, IpcError, OpenMode, QPath};

// ── Linux errno constants ───────────────────────────────────────────

const EBADF: i64 = -9;
const EIO: i64 = -5;
const ENOSYS: i64 = -38;

// ── SyscallBackend trait ────────────────────────────────────────────

/// Abstraction over 9P operations. The Linuxulator calls these to
/// fulfil Linux syscalls. Production implementations wrap the Kernel;
/// tests use [`MockBackend`].
pub trait SyscallBackend {
    fn walk(&mut self, path: &str, new_fid: Fid) -> Result<QPath, IpcError>;
    fn open(&mut self, fid: Fid, mode: OpenMode) -> Result<(), IpcError>;
    fn read(&mut self, fid: Fid, offset: u64, count: u32) -> Result<Vec<u8>, IpcError>;
    fn write(&mut self, fid: Fid, offset: u64, data: &[u8]) -> Result<u32, IpcError>;
    fn clunk(&mut self, fid: Fid) -> Result<(), IpcError>;
}

// ── MockBackend ─────────────────────────────────────────────────────

/// Test double that records all 9P calls for assertion.
#[cfg(test)]
pub struct MockBackend {
    pub walks: Vec<(String, Fid)>,
    pub opens: Vec<(Fid, OpenMode)>,
    pub writes: Vec<(Fid, Vec<u8>)>,
    pub reads: Vec<(Fid, u64, u32)>,
    pub clunks: Vec<Fid>,
}

#[cfg(test)]
impl MockBackend {
    pub fn new() -> Self {
        Self {
            walks: Vec::new(),
            opens: Vec::new(),
            writes: Vec::new(),
            reads: Vec::new(),
            clunks: Vec::new(),
        }
    }
}

#[cfg(test)]
impl SyscallBackend for MockBackend {
    fn walk(&mut self, path: &str, new_fid: Fid) -> Result<QPath, IpcError> {
        self.walks.push((String::from(path), new_fid));
        Ok(0)
    }

    fn open(&mut self, fid: Fid, mode: OpenMode) -> Result<(), IpcError> {
        self.opens.push((fid, mode));
        Ok(())
    }

    fn read(&mut self, fid: Fid, offset: u64, count: u32) -> Result<Vec<u8>, IpcError> {
        self.reads.push((fid, offset, count));
        Ok(Vec::new())
    }

    fn write(&mut self, fid: Fid, offset: u64, data: &[u8]) -> Result<u32, IpcError> {
        self.writes.push((fid, data.to_vec()));
        Ok(data.len() as u32)
    }

    fn clunk(&mut self, fid: Fid) -> Result<(), IpcError> {
        self.clunks.push(fid);
        Ok(())
    }
}
```

Add to `crates/harmony-os/src/lib.rs`:

```rust
pub mod linuxulator;
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os linuxulator::tests::mock_`
Expected: 2 tests PASS

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-os/src/linuxulator.rs crates/harmony-os/src/lib.rs
git commit -m "feat(os): add SyscallBackend trait and MockBackend"
```

---

### Task 4: Linuxulator struct, fd table, and init_stdio

**Files:**
- Modify: `crates/harmony-os/src/linuxulator.rs`

**Context:** The Linuxulator struct owns the fd→fid mapping and a SyscallBackend. `init_stdio()` walks to `/dev/serial/log`, opens it, and maps fd 0/1/2 to serial fids. The Linuxulator allocates fids starting from 100 to avoid collisions with server root fids.

The SerialServer's namespace is: root (fid 0) → "log" (the writable file). So the walk path from the process namespace is `/dev/serial/log` (since SerialServer is mounted at `/dev/serial`).

**Step 1: Write the failing tests**

Add to the tests module in `linuxulator.rs`:

```rust
#[test]
fn linuxulator_init_creates_fd_table() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    lx.init_stdio().unwrap();

    // Should have fd 0, 1, 2
    assert!(lx.has_fd(0));
    assert!(lx.has_fd(1));
    assert!(lx.has_fd(2));
    assert!(!lx.has_fd(3));
}

#[test]
fn init_stdio_walks_and_opens_serial() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    lx.init_stdio().unwrap();

    // Should have walked to /dev/serial/log three times
    // (stdin, stdout, stderr each get their own fid)
    assert_eq!(lx.backend().walks.len(), 3);
    assert_eq!(lx.backend().walks[0].0, "/dev/serial/log");

    // Should have opened all three
    assert_eq!(lx.backend().opens.len(), 3);
}

#[test]
fn linuxulator_starts_not_exited() {
    let mock = MockBackend::new();
    let lx = Linuxulator::new(mock);
    assert!(!lx.exited());
    assert_eq!(lx.exit_code(), None);
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os linuxulator::tests::linuxulator_ linuxulator::tests::init_`
Expected: FAIL — `Linuxulator` doesn't exist

**Step 3: Write the implementation**

Add to `linuxulator.rs`, above the tests module:

```rust
// ── Linuxulator ─────────────────────────────────────────────────────

/// Linux syscall-to-9P translation engine.
///
/// Owns a POSIX-style fd table and dispatches Linux syscalls to a
/// [`SyscallBackend`]. Created once per Linux process.
pub struct Linuxulator<B: SyscallBackend> {
    backend: B,
    /// Maps Linux fd (0, 1, 2, ...) → 9P fid.
    fd_table: BTreeMap<i32, Fid>,
    /// Next fid to allocate for backend calls.
    next_fid: Fid,
    /// Set by sys_exit_group.
    exit_code: Option<i32>,
}

impl<B: SyscallBackend> Linuxulator<B> {
    /// Create a new Linuxulator with an empty fd table.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            fd_table: BTreeMap::new(),
            next_fid: 100, // avoid collision with server root fids
            exit_code: None,
        }
    }

    /// Allocate the next fid for a backend call.
    fn alloc_fid(&mut self) -> Fid {
        let fid = self.next_fid;
        self.next_fid += 1;
        fid
    }

    /// Pre-populate fd 0 (stdin), 1 (stdout), 2 (stderr) by walking
    /// to the serial server and opening the log file.
    ///
    /// Expects SerialServer mounted at `/dev/serial` in the process namespace.
    pub fn init_stdio(&mut self) -> Result<(), IpcError> {
        // stdin (fd 0) — read mode
        let stdin_fid = self.alloc_fid();
        self.backend.walk("/dev/serial/log", stdin_fid)?;
        self.backend.open(stdin_fid, OpenMode::Read)?;
        self.fd_table.insert(0, stdin_fid);

        // stdout (fd 1) — write mode
        let stdout_fid = self.alloc_fid();
        self.backend.walk("/dev/serial/log", stdout_fid)?;
        self.backend.open(stdout_fid, OpenMode::Write)?;
        self.fd_table.insert(1, stdout_fid);

        // stderr (fd 2) — write mode
        let stderr_fid = self.alloc_fid();
        self.backend.walk("/dev/serial/log", stderr_fid)?;
        self.backend.open(stderr_fid, OpenMode::Write)?;
        self.fd_table.insert(2, stderr_fid);

        Ok(())
    }

    /// Check if a Linux fd is in the table.
    pub fn has_fd(&self, fd: i32) -> bool {
        self.fd_table.contains_key(&fd)
    }

    /// Whether the process has called exit_group.
    pub fn exited(&self) -> bool {
        self.exit_code.is_some()
    }

    /// The exit code, if the process has exited.
    pub fn exit_code(&self) -> Option<i32> {
        self.exit_code
    }

    /// Access the backend (for test assertions).
    #[cfg(test)]
    pub fn backend(&self) -> &B {
        &self.backend
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os linuxulator::tests`
Expected: all 5 tests PASS (2 mock + 3 linuxulator)

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-os/src/linuxulator.rs
git commit -m "feat(os): add Linuxulator struct with fd table and stdio init"
```

---

### Task 5: handle_syscall — sys_write and sys_exit_group

**Files:**
- Modify: `crates/harmony-os/src/linuxulator.rs`

**Context:** The core dispatch method. Maps Linux syscall numbers to handler methods. MVP syscalls: `write` (nr 1) and `exit_group` (nr 231). Unknown syscalls return `-ENOSYS`.

`sys_write` reads bytes from the caller's memory via a raw pointer (safe in our flat address space model). For unit tests, we allocate a buffer and pass its address.

**Step 1: Write the failing tests**

```rust
#[test]
fn sys_write_to_stdout() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    lx.init_stdio().unwrap();

    let msg = b"Hello\n";
    let result = lx.handle_syscall(1, [1, msg.as_ptr() as u64, 6, 0, 0, 0]);
    assert_eq!(result, 6); // 6 bytes written

    // Backend should have received the write
    let writes = &lx.backend().writes;
    // Filter to writes after init (init does no writes)
    let stdout_fid = lx.fd_table[&1];
    let stdout_writes: Vec<_> = writes.iter().filter(|(fid, _)| *fid == stdout_fid).collect();
    assert_eq!(stdout_writes.len(), 1);
    assert_eq!(stdout_writes[0].1, b"Hello\n");
}

#[test]
fn sys_write_bad_fd() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    lx.init_stdio().unwrap();

    let msg = b"test";
    let result = lx.handle_syscall(1, [99, msg.as_ptr() as u64, 4, 0, 0, 0]);
    assert_eq!(result, EBADF);
}

#[test]
fn sys_exit_group_sets_flag() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);

    let result = lx.handle_syscall(231, [42, 0, 0, 0, 0, 0]);
    assert_eq!(result, 0);
    assert!(lx.exited());
    assert_eq!(lx.exit_code(), Some(42));
}

#[test]
fn unknown_syscall_returns_enosys() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);

    let result = lx.handle_syscall(9999, [0, 0, 0, 0, 0, 0]);
    assert_eq!(result, ENOSYS);
}

#[test]
fn sys_write_to_stderr() {
    let mock = MockBackend::new();
    let mut lx = Linuxulator::new(mock);
    lx.init_stdio().unwrap();

    let msg = b"err";
    let result = lx.handle_syscall(1, [2, msg.as_ptr() as u64, 3, 0, 0, 0]);
    assert_eq!(result, 3);
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os linuxulator::tests::sys_`
Expected: FAIL — `handle_syscall` doesn't exist

**Step 3: Write the implementation**

Add to the `impl<B: SyscallBackend> Linuxulator<B>` block:

```rust
    /// Dispatch a Linux syscall. Returns the syscall result (negative = errno).
    ///
    /// # Arguments
    /// - `nr`: Linux syscall number (x86_64 ABI)
    /// - `args`: syscall arguments [arg1, arg2, arg3, arg4, arg5, arg6]
    ///
    /// # Safety
    /// For `sys_write`, `args[1]` is treated as a pointer to user memory.
    /// In the MVP flat address space, this is a direct pointer dereference.
    pub fn handle_syscall(&mut self, nr: u64, args: [u64; 6]) -> i64 {
        match nr {
            1 => self.sys_write(args[0] as i32, args[1] as usize, args[2] as usize),
            231 => self.sys_exit_group(args[0] as i32),
            _ => ENOSYS,
        }
    }

    /// Linux write(2): write to a file descriptor.
    fn sys_write(&mut self, fd: i32, buf_ptr: usize, count: usize) -> i64 {
        let fid = match self.fd_table.get(&fd) {
            Some(&fid) => fid,
            None => return EBADF,
        };

        // In the MVP flat address space, we can directly read from the pointer.
        // Safety: caller guarantees buf_ptr points to valid memory of at least
        // `count` bytes. This is the same trust model as a real kernel reading
        // from user space — except here there's no protection boundary.
        let data = unsafe { core::slice::from_raw_parts(buf_ptr as *const u8, count) };

        match self.backend.write(fid, 0, data) {
            Ok(n) => n as i64,
            Err(_) => EIO,
        }
    }

    /// Linux exit_group(2): terminate the process.
    fn sys_exit_group(&mut self, code: i32) -> i64 {
        self.exit_code = Some(code);
        0
    }
```

Also make `fd_table` accessible for the test (add `pub(crate)` or expose via method):

```rust
    /// Look up the fid for a Linux fd (for testing).
    #[cfg(test)]
    pub fn fid_for_fd(&self, fd: i32) -> Option<Fid> {
        self.fd_table.get(&fd).copied()
    }
```

Update the `sys_write_to_stdout` test to use `fid_for_fd` instead of direct field access.

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os linuxulator::tests`
Expected: all 10 tests PASS

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-os/src/linuxulator.rs
git commit -m "feat(os): implement handle_syscall with sys_write and sys_exit_group"
```

---

### Task 6: Integration test — Kernel + SerialServer + Linuxulator

**Files:**
- Modify: `crates/harmony-os/src/linuxulator.rs` (add integration test + KernelBackend)

**Context:** This test wires a real Ring 2 Kernel with a SerialServer, creates a KernelBackend implementing SyscallBackend, and verifies the full pipeline: Linuxulator `sys_write` → KernelBackend → Kernel IPC → SerialServer → buffer contains "Hello\n".

The KernelBackend is `#[cfg(test)]` for now since it requires `std` (Kernel needs harmony-identity). It captures a process PID and delegates all 9P calls to the Kernel.

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use harmony_microkernel::echo::EchoServer;
    use harmony_microkernel::kernel::Kernel;
    use harmony_microkernel::serial_server::SerialServer;
    use harmony_identity::PrivateIdentity;
    use harmony_unikernel::KernelEntropy;

    /// SyscallBackend backed by a real Ring 2 Kernel.
    struct KernelBackend<'a> {
        kernel: &'a mut Kernel,
        pid: u32,
    }

    impl<'a> KernelBackend<'a> {
        fn new(kernel: &'a mut Kernel, pid: u32) -> Self {
            Self { kernel, pid }
        }
    }

    impl SyscallBackend for KernelBackend<'_> {
        fn walk(&mut self, path: &str, new_fid: Fid) -> Result<QPath, IpcError> {
            self.kernel.walk(self.pid, path, 0, new_fid, 0)
        }
        fn open(&mut self, fid: Fid, mode: OpenMode) -> Result<(), IpcError> {
            self.kernel.open(self.pid, fid, mode)
        }
        fn read(&mut self, fid: Fid, offset: u64, count: u32) -> Result<Vec<u8>, IpcError> {
            self.kernel.read(self.pid, fid, offset, count)
        }
        fn write(&mut self, fid: Fid, offset: u64, data: &[u8]) -> Result<u32, IpcError> {
            self.kernel.write(self.pid, fid, offset, data)
        }
        fn clunk(&mut self, fid: Fid) -> Result<(), IpcError> {
            self.kernel.clunk(self.pid, fid)
        }
    }

    fn test_entropy() -> KernelEntropy<impl FnMut(&mut [u8])> {
        let mut seed = 99u64;
        KernelEntropy::new(move |buf: &mut [u8]| {
            for b in buf.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                *b = (seed >> 33) as u8;
            }
        })
    }

    #[test]
    fn linuxulator_writes_hello_through_kernel_to_serial() {
        let mut entropy = test_entropy();
        let kernel_id = PrivateIdentity::generate(&mut entropy);
        let mut kernel = Kernel::new(kernel_id);

        // Spawn SerialServer
        let serial_pid = kernel
            .spawn_process("serial", Box::new(SerialServer::new()), &[])
            .unwrap();

        // Spawn a "linux process" with SerialServer mounted at /dev/serial
        let linux_pid = kernel
            .spawn_process(
                "hello-linux",
                Box::new(EchoServer::new()), // placeholder server
                &[("/dev/serial", serial_pid, 0)],
            )
            .unwrap();

        // Grant the linux process access to the serial server
        kernel
            .grant_endpoint_cap(&mut entropy, linux_pid, serial_pid, 0)
            .unwrap();

        // Create Linuxulator with KernelBackend
        let backend = KernelBackend::new(&mut kernel, linux_pid);
        let mut lx = Linuxulator::new(backend);
        lx.init_stdio().unwrap();

        // Simulate the hello binary's syscalls
        let msg = b"Hello\n";
        let result = lx.handle_syscall(1, [1, msg.as_ptr() as u64, 6, 0, 0, 0]);
        assert_eq!(result, 6);

        // Verify "Hello\n" reached the SerialServer's buffer
        // We need to read back through the kernel to verify
        // Access serial server directly through the process table
        // (The KernelBackend consumed &mut kernel, so we use the backend's reference)
        // Actually, the backend holds &mut kernel. We can read via the linux process.
        let read_fid = 200;
        lx.backend.walk("/dev/serial/log", read_fid).unwrap();
        lx.backend.open(read_fid, OpenMode::Read).unwrap();
        let data = lx.backend.read(read_fid, 0, 256).unwrap();
        assert_eq!(data, b"Hello\n");

        // Verify exit_group
        lx.handle_syscall(231, [0, 0, 0, 0, 0, 0]);
        assert!(lx.exited());
        assert_eq!(lx.exit_code(), Some(0));
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os linuxulator::integration_tests`
Expected: FAIL — module or imports missing

**Step 3: Add necessary imports and make it compile**

The integration test needs `harmony-identity` and `harmony-unikernel` in scope. Since harmony-os depends on harmony-microkernel (which re-exports or provides access to kernel), we need to ensure the test can access `Kernel`, `SerialServer`, etc.

If the Kernel struct isn't accessible, add the necessary dev-dependencies to harmony-os's Cargo.toml:

```toml
[dev-dependencies]
harmony-identity = { workspace = true, features = ["std", "test-utils"] }
harmony-unikernel = { workspace = true, features = ["std"] }
```

The `backend` field also needs to be accessible. Make it `pub(crate)`:

```rust
pub(crate) backend: B,
```

Or add a `pub fn backend_mut(&mut self) -> &mut B` method.

**Step 4: Run test to verify it passes**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-os linuxulator::integration_tests`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-os/src/linuxulator.rs crates/harmony-os/Cargo.toml
git commit -m "test(os): add Linuxulator integration test with Kernel + SerialServer"
```

---

### Task 7: Create test binary (hello.S → hello.elf)

**Files:**
- Create: `crates/harmony-boot/test-bins/hello.S`
- Create: `crates/harmony-boot/test-bins/build.sh`
- Create: `crates/harmony-boot/test-bins/hello.elf` (pre-built binary)

**Context:** A minimal x86_64 Linux binary that calls `write(1, "Hello\n", 6)` then `exit_group(0)`. Cross-compiled with GNU binutils for x86_64. The pre-built ELF is committed so builds work without a cross-compiler.

**Step 1: Write hello.S**

Create `crates/harmony-boot/test-bins/hello.S`:

```asm
# Minimal x86_64 Linux binary for Linuxulator MVP testing.
# Calls write(1, "Hello\n", 6) then exit_group(0).
#
# Build: x86_64-linux-gnu-as -o hello.o hello.S && x86_64-linux-gnu-ld -o hello.elf -nostdlib --static hello.o
# Or on Linux: as -o hello.o hello.S && ld -o hello.elf -nostdlib --static hello.o

.globl _start
_start:
    # write(1, msg, 6)
    mov $1, %rax            # syscall: write
    mov $1, %rdi            # fd: stdout
    lea msg(%rip), %rsi     # buf: address of message
    mov $6, %rdx            # count: 6 bytes
    syscall

    # exit_group(0)
    mov $231, %rax          # syscall: exit_group
    xor %rdi, %rdi          # code: 0
    syscall

msg:
    .ascii "Hello\n"
```

**Step 2: Write build.sh**

Create `crates/harmony-boot/test-bins/build.sh`:

```bash
#!/bin/sh
# Build the test ELF binary. Requires x86_64 GNU binutils.
# On macOS: brew install x86_64-elf-binutils
# On Linux: apt install binutils
set -e

cd "$(dirname "$0")"

# Try platform-specific assembler names
AS="${AS:-}"
LD="${LD:-}"

if [ -z "$AS" ]; then
    if command -v x86_64-linux-gnu-as >/dev/null 2>&1; then
        AS=x86_64-linux-gnu-as
        LD=x86_64-linux-gnu-ld
    elif command -v x86_64-elf-as >/dev/null 2>&1; then
        AS=x86_64-elf-as
        LD=x86_64-elf-ld
    elif command -v as >/dev/null 2>&1; then
        AS=as
        LD=ld
    else
        echo "No x86_64 assembler found. Install x86_64-elf-binutils." >&2
        exit 1
    fi
fi

$AS -o hello.o hello.S
$LD -o hello.elf -nostdlib --static hello.o
rm -f hello.o
echo "Built hello.elf ($(wc -c < hello.elf | tr -d ' ') bytes)"
```

**Step 3: Build and commit the ELF**

```bash
cd crates/harmony-boot/test-bins
chmod +x build.sh
./build.sh   # or cross-compile manually
```

If you can't build locally (no x86_64 assembler on macOS), install the toolchain:

```bash
brew install x86_64-elf-binutils
```

Then build with `AS=x86_64-elf-as LD=x86_64-elf-ld ./build.sh`.

**Step 4: Verify the ELF parses correctly**

Add a quick test in harmony-os that loads the real binary (if accessible):

```rust
// In elf.rs tests — optional, only if the binary is accessible
#[test]
fn parse_real_hello_elf() {
    let elf_bytes = include_bytes!("../../../harmony-boot/test-bins/hello.elf");
    let parsed = parse_elf(elf_bytes).unwrap();
    assert_eq!(parsed.segments.len(), 1);
    assert!(parsed.segments[0].flags.execute);
}
```

Note: This test may not work if harmony-boot is outside the workspace path. If so, skip it — the synthetic ELF tests in Task 2 already verify the parser.

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-boot/test-bins/hello.S crates/harmony-boot/test-bins/build.sh crates/harmony-boot/test-bins/hello.elf
git commit -m "feat(boot): add hello.S test binary for Linuxulator MVP"
```

---

### Task 8: harmony-boot Ring 3 integration (MSR setup, syscall handler, QEMU)

**Files:**
- Create: `crates/harmony-boot/src/syscall.rs`
- Modify: `crates/harmony-boot/src/main.rs` (add `ring3` block)
- Modify: `crates/harmony-boot/Cargo.toml` (add `ring3` feature + harmony-os dep)

**Context:** This is the bare-metal integration. Sets up x86_64 MSRs so the `syscall` instruction traps to our handler. Loads the ELF from `include_bytes!`, creates a Linuxulator with a DirectBackend (wrapping SerialServer directly — the Kernel requires `std` which isn't available bare-metal), and jumps to the binary's entry point.

The DirectBackend bypasses the Kernel and calls SerialServer methods directly. This still exercises the 9P FileServer interface — it just skips the capability/namespace layer (which requires `std` for identity stores).

**Step 1: Update Cargo.toml**

Add to `crates/harmony-boot/Cargo.toml`:

```toml
harmony-os = { path = "../harmony-os", default-features = false, optional = true }

[features]
ring3 = ["ring2", "dep:harmony-os"]
```

**Step 2: Create syscall.rs**

Create `crates/harmony-boot/src/syscall.rs`:

```rust
//! x86_64 syscall trap setup for the Linuxulator.
//!
//! Configures MSRs so the `syscall` instruction transfers control to
//! our handler, which dispatches to the Linuxulator.

use core::arch::asm;

// ── MSR addresses ───────────────────────────────────────────────────

const IA32_EFER: u32 = 0xC000_0080;
const IA32_STAR: u32 = 0xC000_0081;
const IA32_LSTAR: u32 = 0xC000_0082;
const IA32_FMASK: u32 = 0xC000_0084;

const EFER_SCE: u64 = 1; // System Call Enable

/// Read a Model-Specific Register.
unsafe fn rdmsr(msr: u32) -> u64 {
    let (low, high): (u32, u32);
    asm!(
        "rdmsr",
        in("ecx") msr,
        out("eax") low,
        out("edx") high,
        options(nomem, nostack, preserves_flags),
    );
    (high as u64) << 32 | low as u64
}

/// Write a Model-Specific Register.
unsafe fn wrmsr(msr: u32, value: u64) {
    let low = value as u32;
    let high = (value >> 32) as u32;
    asm!(
        "wrmsr",
        in("ecx") msr,
        in("eax") low,
        in("edx") high,
        options(nomem, nostack, preserves_flags),
    );
}

// ── Syscall dispatch ────────────────────────────────────────────────

/// Result of a syscall dispatch. Includes the return value and whether
/// the process called exit_group.
pub struct SyscallResult {
    pub retval: i64,
    pub exited: bool,
    pub exit_code: i32,
}

/// Function pointer type for the syscall dispatcher.
/// Called from the naked handler with syscall number and arguments.
pub type SyscallDispatchFn = fn(nr: u64, args: [u64; 6]) -> SyscallResult;

/// Global dispatch function. Set before enabling syscalls.
static mut DISPATCH_FN: Option<SyscallDispatchFn> = None;

/// Whether the Linux process has exited.
static mut PROCESS_EXITED: bool = false;
static mut EXIT_CODE: i32 = 0;

/// Install the syscall dispatch function.
///
/// # Safety
/// Must be called before `setup_msrs` and before any `syscall` executes.
pub unsafe fn set_dispatch_fn(f: SyscallDispatchFn) {
    DISPATCH_FN = Some(f);
}

/// Check if the process exited after a syscall.
pub fn process_exited() -> bool {
    unsafe { PROCESS_EXITED }
}

/// Get the exit code.
pub fn exit_code() -> i32 {
    unsafe { EXIT_CODE }
}

/// The Rust-side syscall handler. Called from the naked assembly trampoline.
///
/// # Safety
/// Called from assembly with the correct register layout.
#[no_mangle]
unsafe extern "C" fn rust_syscall_handler(
    nr: u64,
    arg1: u64,
    arg2: u64,
    arg3: u64,
    arg4: u64,
    arg5: u64,
    arg6: u64,
) -> i64 {
    if let Some(dispatch) = DISPATCH_FN {
        let result = dispatch(nr, [arg1, arg2, arg3, arg4, arg5, arg6]);
        if result.exited {
            PROCESS_EXITED = true;
            EXIT_CODE = result.exit_code;
        }
        result.retval
    } else {
        -38 // ENOSYS
    }
}

// ── Naked syscall entry point ───────────────────────────────────────

/// The raw syscall entry point. The CPU jumps here on `syscall`.
///
/// Register state on entry (Linux x86_64 ABI):
///   RAX = syscall number
///   RDI = arg1, RSI = arg2, RDX = arg3
///   R10 = arg4, R8 = arg5, R9 = arg6
///   RCX = return RIP (saved by CPU)
///   R11 = return RFLAGS (saved by CPU)
#[naked]
#[no_mangle]
unsafe extern "C" fn syscall_entry() {
    asm!(
        // Save registers that sysretq needs and callee-saved regs
        "push rcx",          // return RIP
        "push r11",          // return RFLAGS
        "push rbx",
        "push rbp",
        "push r12",
        "push r13",
        "push r14",
        "push r15",

        // Set up arguments for rust_syscall_handler(nr, a1, a2, a3, a4, a5, a6)
        // SysV calling convention: rdi, rsi, rdx, rcx, r8, r9, [stack]
        //
        // Current:  RAX=nr, RDI=a1, RSI=a2, RDX=a3, R10=a4, R8=a5, R9=a6
        // Need:     RDI=nr, RSI=a1, RDX=a2, RCX=a3, R8=a4, R9=a5, [stack]=a6
        //
        // Shuffle carefully to avoid clobbering:
        "push r9",           // save a6 for stack arg
        "mov r9, r8",        // r9 = a5
        "mov r8, r10",       // r8 = a4
        "mov rcx, rdx",      // rcx = a3
        "mov rdx, rsi",      // rdx = a2
        "mov rsi, rdi",      // rsi = a1
        "mov rdi, rax",      // rdi = nr

        // Push a6 as 7th arg (stack)
        // It's already on the stack from "push r9" above.
        // But we need proper stack alignment. The 8 pushes + 1 push = 9*8 = 72 bytes.
        // For 16-byte alignment before call: 72 is not 16-aligned, so push one more.
        "sub rsp, 8",        // align stack to 16 bytes

        "call rust_syscall_handler",

        // Return value is in RAX
        "add rsp, 16",       // pop alignment padding + a6

        // Restore callee-saved registers
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbp",
        "pop rbx",
        "pop r11",           // return RFLAGS
        "pop rcx",           // return RIP

        "sysretq",
        options(noreturn),
    );
}

// ── MSR setup ───────────────────────────────────────────────────────

/// Configure x86_64 MSRs for syscall interception.
///
/// # Safety
/// Must be called in Ring 0 with interrupts disabled.
/// The GDT must have a valid kernel code segment.
pub unsafe fn setup_msrs(kernel_cs: u16) {
    // Enable System Call Enable bit in EFER
    let efer = rdmsr(IA32_EFER);
    wrmsr(IA32_EFER, efer | EFER_SCE);

    // LSTAR — syscall entry point
    wrmsr(IA32_LSTAR, syscall_entry as u64);

    // STAR — kernel CS in bits 47:32
    // On syscall: CS = STAR[47:32], SS = STAR[47:32] + 8
    // On sysret:  CS = STAR[63:48] + 16, SS = STAR[63:48] + 8
    // We only care about the syscall side for MVP (everything is Ring 0).
    let star = (kernel_cs as u64) << 32;
    wrmsr(IA32_STAR, star);

    // FMASK — clear IF (bit 9) on syscall entry to prevent interrupts
    wrmsr(IA32_FMASK, 0x200);
}
```

**Step 3: Add ring3 block to main.rs**

Add after the existing `#[cfg(feature = "ring2")]` block in `kernel_main()`:

```rust
    // ── Ring 3 Linuxulator ────────────────────────────────────────────
    #[cfg(feature = "ring3")]
    {
        use harmony_microkernel::serial_server::SerialServer as MkSerialServer;
        use harmony_microkernel::FileServer;
        use harmony_os::elf::parse_elf;
        use harmony_os::linuxulator::{Linuxulator, SyscallBackend};

        serial.log("KERN", "Ring 3 Linuxulator mode");

        // DirectBackend wraps a SerialServer directly (no Kernel needed for bare-metal)
        struct DirectBackend {
            server: MkSerialServer,
            stdout_fid_walked: bool,
        }

        impl DirectBackend {
            fn new() -> Self {
                Self {
                    server: MkSerialServer::new(),
                    stdout_fid_walked: false,
                }
            }
        }

        impl SyscallBackend for DirectBackend {
            fn walk(&mut self, path: &str, new_fid: harmony_microkernel::Fid) -> Result<harmony_microkernel::QPath, harmony_microkernel::IpcError> {
                // All walks go to "log" — the only file in SerialServer
                self.server.walk(0, new_fid, "log")
            }
            fn open(&mut self, fid: harmony_microkernel::Fid, mode: harmony_microkernel::OpenMode) -> Result<(), harmony_microkernel::IpcError> {
                self.server.open(fid, mode)
            }
            fn read(&mut self, fid: harmony_microkernel::Fid, offset: u64, count: u32) -> Result<alloc::vec::Vec<u8>, harmony_microkernel::IpcError> {
                self.server.read(fid, offset, count)
            }
            fn write(&mut self, fid: harmony_microkernel::Fid, offset: u64, data: &[u8]) -> Result<u32, harmony_microkernel::IpcError> {
                // Write to SerialServer buffer AND echo to real serial port
                let result = self.server.write(fid, offset, data);
                // Also write to actual serial for QEMU visibility
                for &byte in data {
                    serial_write_byte(byte);
                }
                result
            }
            fn clunk(&mut self, fid: harmony_microkernel::Fid) -> Result<(), harmony_microkernel::IpcError> {
                self.server.clunk(fid)
            }
        }

        // 1. Load ELF
        let elf_bytes = include_bytes!("../../test-bins/hello.elf");
        let parsed = match parse_elf(elf_bytes) {
            Ok(p) => p,
            Err(e) => {
                let _ = writeln!(serial, "[LINUX] ELF parse error: {:?}", e);
                loop { x86_64::instructions::hlt(); }
            }
        };
        let _ = writeln!(serial, "[LINUX] loaded hello.elf ({} bytes, {} segments)",
            elf_bytes.len(), parsed.segments.len());
        let _ = writeln!(serial, "[LINUX] entry=0x{:x}", parsed.entry_point);

        // 2. Copy segment to heap
        let seg = &parsed.segments[0];
        let mem = alloc::vec![0u8; seg.memsz as usize];
        let mem_ptr = mem.as_ptr();
        let filesz = seg.filesz as usize;
        unsafe {
            core::ptr::copy_nonoverlapping(
                elf_bytes.as_ptr().add(seg.offset as usize),
                mem_ptr as *mut u8,
                filesz,
            );
        }
        // The entry point offset within the segment
        let entry_offset = parsed.entry_point - seg.vaddr;
        let real_entry = unsafe { mem_ptr.add(entry_offset as usize) };
        let _ = writeln!(serial, "[LINUX] segment at {:p}, entry at {:p}", mem_ptr, real_entry);

        // 3. Allocate stack
        let stack_size = 64 * 1024; // 64 KiB
        let stack = alloc::vec![0u8; stack_size];
        let stack_top = unsafe { stack.as_ptr().add(stack_size) as u64 };

        // 4. Create Linuxulator
        let backend = DirectBackend::new();
        let mut linuxulator = Linuxulator::new(backend);
        linuxulator.init_stdio().expect("init_stdio failed");

        // 5. Set up syscall dispatch
        fn dispatch(nr: u64, args: [u64; 6]) -> crate::syscall::SyscallResult {
            // Safety: single-threaded, no concurrency
            static mut LX: Option<*mut Linuxulator<DirectBackend>> = None;
            // This is set up before the binary runs
            // For MVP: use the global linuxulator
            todo!("wire global dispatch — see step 6")
        }

        // 6. Set up MSRs and jump
        // Note: The exact global-state wiring for the dispatch function
        // requires careful lifetime management. The implementer should:
        // a) Store the Linuxulator in a static mut
        // b) Set DISPATCH_FN to a function that accesses it
        // c) Set up MSRs
        // d) Set RSP to stack_top, jump to real_entry
        // e) On return (after exit_group triggers sysretq back to caller),
        //    log the exit code
        //
        // The jump to the binary is an inline asm block:
        //   mov rsp, stack_top
        //   jmp real_entry
        //
        // After exit_group, the syscall handler should NOT sysretq.
        // Instead it should return to kernel_main (e.g., via longjmp-style
        // or by checking the exit flag after each syscall and branching).

        serial.log("LINUX", "ring3 integration TBD — see Task 8 notes");
    }
```

**Important implementer notes for this task:**

The global dispatch wiring is the trickiest part. The recommended approach:

1. Use a `static mut LINUXULATOR_PTR: *mut u8 = core::ptr::null_mut()` to store a type-erased pointer to the Linuxulator.
2. The dispatch function casts it back and calls `handle_syscall`.
3. After `syscall` + `sysretq` returns to the binary, the binary executes `exit_group` which sets the exit flag. The syscall handler checks the flag — instead of `sysretq`, it restores the original RSP and returns to `kernel_main`.
4. Alternatively: the syscall handler for `exit_group` can just `hlt` (for MVP) or do a QEMU debug exit.

The implementer should adapt this skeleton based on what actually works. The core logic (ELF parsing, Linuxulator dispatch, 9P writes) is already tested by Tasks 2-6. This task is about the bare-metal glue.

**Step 4: Build for bare-metal**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os/crates/harmony-boot
cargo check --target x86_64-unknown-none --features ring3
```

Expected: compiles (with warnings about the `todo!()`)

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-boot/src/syscall.rs crates/harmony-boot/src/main.rs crates/harmony-boot/Cargo.toml
git commit -m "feat(boot): add Ring 3 Linuxulator integration with x86_64 syscall trap"
```

---

### Task 9: Quality gates

**Files:** All modified files

**Step 1: Run full test suite (harmony-os workspace)**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test --workspace`
Expected: all tests PASS (29 existing + ~18 new)

**Step 2: Run clippy**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo clippy --workspace`
Expected: zero errors (warnings acceptable for the `todo!()` in boot integration)

**Step 3: Run fmt**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo fmt --all -- --check`
Expected: no diffs. Fix with `cargo fmt --all` if needed.

**Step 4: Verify no_std compilation**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo check -p harmony-os --no-default-features`
Expected: compiles

**Step 5: Verify test count**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test --workspace 2>&1 | grep "test result"`
Expected: all suites show 0 failed

**Step 6: Commit any fixes**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add -A
git commit -m "style: apply cargo fmt and fix clippy warnings"
```

---

## Summary

| Task | What | Tests | Repo |
|------|------|-------|------|
| 1 | no_std feature gating for harmony-os | compile check | harmony-os |
| 2 | ELF parser (elf.rs) | 8 tests | harmony-os |
| 3 | SyscallBackend trait + MockBackend | 2 tests | harmony-os |
| 4 | Linuxulator struct, fd table, init_stdio | 3 tests | harmony-os |
| 5 | handle_syscall (sys_write, sys_exit_group) | 5 tests | harmony-os |
| 6 | Integration test (Kernel + SerialServer + Linuxulator) | 1 test | harmony-os |
| 7 | Test binary (hello.S → hello.elf) | build verification | harmony-os |
| 8 | harmony-boot ring3 (MSR, naked asm, QEMU) | QEMU boot test | harmony-os |
| 9 | Quality gates | full suite | harmony-os |

**Total new tests:** ~19 unit/integration + 1 QEMU integration

**Critical path:** Tasks 1-6 are fully testable without bare metal. Task 7-8 require x86_64 cross-compilation and QEMU. Task 8 is the most complex (global state, naked asm, jump-to-ELF) and may need iteration.

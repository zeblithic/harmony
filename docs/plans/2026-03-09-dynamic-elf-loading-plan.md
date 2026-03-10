# Dynamic ELF Loading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable dynamically-linked aarch64 musl binaries to run on Harmony by extending the ELF parser, adding an ElfLoader trait, implementing file-backed mmap, and serving libraries from a virtual /lib namespace.

**Architecture:** Kernel parses PT_INTERP, loads interpreter (ld-musl) + executable into memory, constructs auxiliary vector and initial stack, jumps to interpreter's entry point. ld-musl handles all dynamic linking (relocations, symbol resolution, .so loading) via standard syscalls (open, mmap, mprotect). Libraries served from a 9P LibraryServer mounted at /lib.

**Tech Stack:** Rust (no_std), harmony-os Ring 3 (Linuxulator), harmony-microkernel Ring 2 (FileServer, 9P IPC), ELF64 format, musl libc

**Design doc:** `docs/plans/2026-03-09-dynamic-elf-loading-design.md`

**Repos:** All code changes in `harmony-os`. Design docs in `harmony` (this repo).

---

### Task 1: ELF Parser — Accept ET_DYN Binaries

The ELF parser currently rejects ET_DYN (shared libraries and PIE executables). Dynamic binaries and the interpreter itself are ET_DYN. We need to accept both ET_EXEC and ET_DYN.

**Files:**
- Modify: `crates/harmony-os/src/elf.rs`

**Step 1: Write the failing tests**

In `crates/harmony-os/src/elf.rs`, inside the `mod tests` block, modify the existing `reject_dynamic_elf` test and add new tests:

```rust
#[test]
fn accept_et_dyn() {
    let code = [0xCC; 16];
    let mut elf = build_test_elf(&code);
    // Change e_type from ET_EXEC (2) to ET_DYN (3)
    elf[16..18].copy_from_slice(&3u16.to_le_bytes());
    let parsed = parse_elf(&elf).unwrap();
    assert_eq!(parsed.entry_point, 0x401000);
    assert_eq!(parsed.segments.len(), 1);
}

#[test]
fn accept_et_exec() {
    // Existing static binaries still work
    let code = [0xCC; 16];
    let elf = build_test_elf(&code);
    let parsed = parse_elf(&elf).unwrap();
    assert_eq!(parsed.entry_point, 0x401000);
}

#[test]
fn reject_et_rel() {
    let mut elf = build_test_elf(&[0xCC]);
    elf[16..18].copy_from_slice(&1u16.to_le_bytes()); // ET_REL (relocatable)
    assert_eq!(parse_elf(&elf), Err(ElfError::NotExecutable));
}

#[test]
fn et_dyn_reports_elf_type() {
    let code = [0xCC; 16];
    let mut elf = build_test_elf(&code);
    elf[16..18].copy_from_slice(&3u16.to_le_bytes());
    let parsed = parse_elf(&elf).unwrap();
    assert_eq!(parsed.elf_type, ElfType::Dyn);
}

#[test]
fn et_exec_reports_elf_type() {
    let code = [0xCC; 16];
    let elf = build_test_elf(&code);
    let parsed = parse_elf(&elf).unwrap();
    assert_eq!(parsed.elf_type, ElfType::Exec);
}
```

Remove the old `reject_dynamic_elf` test (line 340).

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-os -- elf::tests`
Expected: FAIL — `ElfType` doesn't exist, `elf_type` field doesn't exist, `accept_et_dyn` gets `NotExecutable`

**Step 3: Implement**

In `crates/harmony-os/src/elf.rs`:

1. Add `ET_DYN` constant next to `ET_EXEC` (line 16):
```rust
const ET_DYN: u16 = 3;
```

2. Add `ElfType` enum after the error type (around line 44):
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElfType {
    Exec,
    Dyn,
}
```

3. Add `elf_type` field to `ParsedElf` (line 74):
```rust
pub struct ParsedElf {
    pub entry_point: u64,
    pub elf_type: ElfType,
    pub segments: Vec<ElfSegment>,
}
```

4. Change the `e_type` check in `parse_elf()` (lines 133-137):
```rust
let e_type = u16_le(data, 16);
let elf_type = match e_type {
    ET_EXEC => ElfType::Exec,
    ET_DYN => ElfType::Dyn,
    _ => return Err(ElfError::NotExecutable),
};
```

5. Add `elf_type` to the `Ok(ParsedElf { ... })` return (line 221):
```rust
Ok(ParsedElf {
    entry_point,
    elf_type,
    segments,
})
```

6. Update the module doc comment (lines 3-7) to mention ET_DYN support.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-os -- elf::tests`
Expected: PASS (all existing + new tests)

Note: If any existing tests construct `ParsedElf` directly, they'll need the new `elf_type` field added.

**Step 5: Commit**

```
git add crates/harmony-os/src/elf.rs
git commit -m "feat(elf): accept ET_DYN binaries alongside ET_EXEC"
```

---

### Task 2: ELF Parser — PT_INTERP and Phdr Metadata

Parse the PT_INTERP segment to extract the interpreter path, and expose program header metadata for auxiliary vector construction.

**Files:**
- Modify: `crates/harmony-os/src/elf.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn parse_pt_interp() {
    let interp_path = b"/lib/ld-musl-aarch64.so.1\0";
    let elf = build_test_elf_with_interp(&[0xCC; 16], interp_path);
    let parsed = parse_elf(&elf).unwrap();
    assert_eq!(
        parsed.interpreter.as_deref(),
        Some("/lib/ld-musl-aarch64.so.1")
    );
}

#[test]
fn no_interp_for_static_elf() {
    let code = [0xCC; 16];
    let elf = build_test_elf(&code);
    let parsed = parse_elf(&elf).unwrap();
    assert!(parsed.interpreter.is_none());
}

#[test]
fn phdr_metadata_present() {
    let code = [0xCC; 16];
    let elf = build_test_elf(&code);
    let parsed = parse_elf(&elf).unwrap();
    assert_eq!(parsed.phdr_offset, 64);
    assert_eq!(parsed.phdr_entry_size, 56);
    assert_eq!(parsed.phdr_count, 1);
}

#[test]
fn interp_with_phdr_metadata() {
    let interp_path = b"/lib/ld-musl-aarch64.so.1\0";
    let elf = build_test_elf_with_interp(&[0xCC; 16], interp_path);
    let parsed = parse_elf(&elf).unwrap();
    assert_eq!(parsed.phdr_count, 2); // PT_LOAD + PT_INTERP
}

#[test]
fn invalid_interp_path() {
    let mut elf = build_test_elf_with_interp(&[0xCC; 16], b"/lib/ld\0");
    // Corrupt the PT_INTERP offset to point past the file
    let interp_ph_start = 64 + 56; // second phdr
    elf[interp_ph_start + 8..interp_ph_start + 16]
        .copy_from_slice(&(elf.len() as u64 + 100).to_le_bytes());
    assert_eq!(parse_elf(&elf), Err(ElfError::InterpreterPathInvalid));
}
```

Also add the test helper `build_test_elf_with_interp`:

```rust
/// Build a test ELF with a PT_INTERP segment in addition to PT_LOAD.
fn build_test_elf_with_interp(code: &[u8], interp: &[u8]) -> Vec<u8> {
    let phdr_size = 56;
    let phnum = 2;
    let code_offset = 64 + phnum * phdr_size;
    let interp_offset = code_offset + code.len();
    let total = interp_offset + interp.len();
    let mut elf = vec![0u8; total];

    // ELF header
    elf[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    elf[4] = 2; // ELFCLASS64
    elf[5] = 1; // ELFDATA2LSB
    elf[6] = 1; // EV_CURRENT
    elf[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    #[cfg(target_arch = "x86_64")]
    elf[18..20].copy_from_slice(&0x3Eu16.to_le_bytes());
    #[cfg(target_arch = "aarch64")]
    elf[18..20].copy_from_slice(&0xB7u16.to_le_bytes());
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    elf[18..20].copy_from_slice(&0x3Eu16.to_le_bytes());
    elf[20..24].copy_from_slice(&1u32.to_le_bytes());
    elf[24..32].copy_from_slice(&0x401000u64.to_le_bytes());
    elf[32..40].copy_from_slice(&64u64.to_le_bytes()); // e_phoff
    elf[52..54].copy_from_slice(&64u16.to_le_bytes());
    elf[54..56].copy_from_slice(&56u16.to_le_bytes());
    elf[56..58].copy_from_slice(&(phnum as u16).to_le_bytes());

    // PT_LOAD program header
    let ph = &mut elf[64..64 + phdr_size];
    ph[0..4].copy_from_slice(&1u32.to_le_bytes()); // PT_LOAD
    ph[4..8].copy_from_slice(&5u32.to_le_bytes()); // PF_R | PF_X
    ph[8..16].copy_from_slice(&(code_offset as u64).to_le_bytes());
    ph[16..24].copy_from_slice(&0x401000u64.to_le_bytes());
    ph[24..32].copy_from_slice(&0x401000u64.to_le_bytes());
    ph[32..40].copy_from_slice(&(code.len() as u64).to_le_bytes());
    ph[40..48].copy_from_slice(&(code.len() as u64).to_le_bytes());
    ph[48..56].copy_from_slice(&0x1000u64.to_le_bytes());

    // PT_INTERP program header
    let pt_interp: u32 = 3;
    let iph = &mut elf[64 + phdr_size..64 + 2 * phdr_size];
    iph[0..4].copy_from_slice(&pt_interp.to_le_bytes());
    iph[4..8].copy_from_slice(&4u32.to_le_bytes()); // PF_R
    iph[8..16].copy_from_slice(&(interp_offset as u64).to_le_bytes());
    iph[32..40].copy_from_slice(&(interp.len() as u64).to_le_bytes());
    iph[40..48].copy_from_slice(&(interp.len() as u64).to_le_bytes());
    iph[48..56].copy_from_slice(&1u64.to_le_bytes());

    elf[code_offset..code_offset + code.len()].copy_from_slice(code);
    elf[interp_offset..interp_offset + interp.len()].copy_from_slice(interp);

    elf
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-os -- elf::tests`
Expected: FAIL — `interpreter` field doesn't exist, `phdr_offset` etc. don't exist

**Step 3: Implement**

1. Add `PT_INTERP` constant (near line 21):
```rust
const PT_INTERP: u32 = 3;
```

2. Add `InterpreterPathInvalid` to `ElfError` enum.

3. Add fields to `ParsedElf`:
```rust
pub struct ParsedElf {
    pub entry_point: u64,
    pub elf_type: ElfType,
    pub segments: Vec<ElfSegment>,
    pub interpreter: Option<String>,
    pub phdr_offset: u64,
    pub phdr_entry_size: u16,
    pub phdr_count: u16,
}
```

4. In `parse_elf()`, after parsing PT_LOAD segments, add PT_INTERP parsing:
```rust
let mut interpreter = None;
for i in 0..e_phnum {
    let ph = &data[e_phoff + i * e_phentsize..];
    let p_type = u32_le(ph, 0);
    if p_type != PT_INTERP {
        continue;
    }
    let p_offset = u64_le(ph, 8) as usize;
    let p_filesz = u64_le(ph, 32) as usize;
    if p_offset.checked_add(p_filesz).map_or(true, |end| end > data.len()) {
        return Err(ElfError::InterpreterPathInvalid);
    }
    let interp_bytes = &data[p_offset..p_offset + p_filesz];
    let path = interp_bytes.strip_suffix(&[0]).unwrap_or(interp_bytes);
    let path_str = core::str::from_utf8(path)
        .map_err(|_| ElfError::InterpreterPathInvalid)?;
    interpreter = Some(String::from(path_str));
    break;
}
```

5. Update the return with all new fields. Add `use alloc::string::String;`.

**Step 4: Run tests**

Run: `cargo test -p harmony-os -- elf::tests`
Expected: PASS

**Step 5: Commit**

```
git add crates/harmony-os/src/elf.rs
git commit -m "feat(elf): parse PT_INTERP and expose phdr metadata"
```

---

### Task 3: LibraryServer — Virtual /lib FileServer

Create a read-only 9P FileServer that maps library filenames to raw ELF bytes.

**Files:**
- Create: `crates/harmony-microkernel/src/library_server.rs`
- Modify: `crates/harmony-microkernel/src/lib.rs` (add module)

**Step 1: Write the failing tests**

Create `crates/harmony-microkernel/src/library_server.rs` with test module containing tests for: walk existing/nonexistent library, open+read, read with offset, read past EOF, stat returns size, clunk releases fid, write rejected, read without open fails.

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-microkernel -- library_server::tests`
Expected: FAIL — `LibraryServer` doesn't exist

**Step 3: Implement**

Implement `LibraryServer` with:
- `manifest: BTreeMap<Arc<str>, Vec<u8>>` mapping library names to bytes
- `fids: BTreeMap<Fid, FidState>` tracking walk/open state per fid
- `FileServer` impl: walk (lookup in manifest), open (mark opened), read (slice bytes), write (return ReadOnly), clunk (remove fid), stat (return file size)

Add `pub mod library_server;` to `crates/harmony-microkernel/src/lib.rs`. Check whether it needs `#[cfg(feature = "kernel")]` gating by looking at what `content_server` uses.

**Step 4: Run tests**

Run: `cargo test -p harmony-microkernel -- library_server::tests`
Expected: PASS

**Step 5: Commit**

```
git add crates/harmony-microkernel/src/library_server.rs crates/harmony-microkernel/src/lib.rs
git commit -m "feat(microkernel): add LibraryServer for virtual /lib namespace"
```

---

### Task 4: New Syscall Variants — Writev, Lseek, Getrandom, Getcwd, Readlink

ld-musl's startup needs these syscalls. Add variants, mappings, and implementations.

**Files:**
- Modify: `crates/harmony-os/src/linuxulator.rs`

**Step 1: Write the failing tests**

Add tests for: `sys_writev_single_iovec`, `sys_writev_multiple_iovecs`, `sys_lseek_set`, `sys_lseek_bad_fd`, `sys_getrandom_fills_buffer`, `sys_getcwd_returns_root`, `sys_readlink_returns_enosys`.

Use `handle_syscall` with the x86_64 syscall numbers: writev=20, lseek=8, getcwd=79, readlink=89, getrandom=318.

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-os -- linuxulator::tests`
Expected: FAIL — syscall numbers map to `Unknown`, return ENOSYS

**Step 3: Implement**

1. Add `LinuxSyscall` variants: `Writev { fd, iov, iovcnt }`, `Lseek { fd, offset, whence }`, `Getrandom { buf, buflen, flags }`, `Getcwd { buf, size }`, `Readlink { pathname, buf, bufsiz }`

2. Add x86_64 mappings: 8→Lseek, 20→Writev, 79→Getcwd, 89→Readlink, 318→Getrandom

3. Add aarch64 mappings: 17→Getcwd, 62→Lseek, 66→Writev, 78→Readlink, 278→Getrandom

4. Add dispatch arms and handler methods:
   - `sys_writev`: iterate iovecs (16 bytes each: base+len), delegate each to `sys_write`
   - `sys_lseek`: SEEK_SET/SEEK_CUR update `FdEntry.offset`
   - `sys_getrandom`: deterministic fill using address-based hash pattern
   - `sys_getcwd`: write `"/\0"` to buffer, return buffer pointer
   - `Readlink`: dispatch to ENOSYS directly

**Step 4: Run tests**

Run: `cargo test -p harmony-os -- linuxulator::tests`
Expected: PASS

**Step 5: Commit**

```
git add crates/harmony-os/src/linuxulator.rs
git commit -m "feat(linuxulator): add writev, lseek, getrandom, getcwd, readlink syscalls"
```

---

### Task 5: File-Backed mmap

Extend `sys_mmap` to handle non-anonymous mappings by reading file content from the backend.

**Files:**
- Modify: `crates/harmony-os/src/linuxulator.rs`

**Step 1: Write the failing tests**

Test that mmap with a valid fd (not MAP_ANONYMOUS) succeeds instead of returning EINVAL. Test both arena fallback and VM-backed paths.

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-os -- linuxulator::tests`
Expected: FAIL — returns EINVAL because `flags & MAP_ANONYMOUS == 0`

**Step 3: Implement**

1. Remove the `MAP_ANONYMOUS` requirement check (lines 1164-1166)
2. Rename `_fd` → `fd` and `_offset` → `offset` in `sys_mmap` signature
3. For file-backed mmap: look up fd in fd_table, read file content via `backend.read()`, allocate pages, copy data into mapped region
4. Arena path: allocate anonymous memory, copy file content in
5. VM path: allocate via `vm_mmap`, copy data, then `vm_mprotect` to final permissions (pages need to be writable during copy, then set to requested prot)

**Step 4: Run tests**

Run: `cargo test -p harmony-os -- linuxulator::tests`
Expected: PASS

**Step 5: Commit**

```
git add crates/harmony-os/src/linuxulator.rs
git commit -m "feat(linuxulator): add file-backed mmap support"
```

---

### Task 6: ElfLoader Trait and InterpreterLoader

Create the `ElfLoader` trait and `InterpreterLoader` that loads executable + interpreter, building the auxiliary vector.

**Files:**
- Create: `crates/harmony-os/src/elf_loader.rs`
- Modify: `crates/harmony-os/src/lib.rs` (add module)

**Step 1: Write the failing tests**

Tests for: `load_static_elf_returns_exe_entry`, `load_dynamic_elf_returns_interp_entry`, `auxv_contains_required_entries`, `et_dyn_loaded_at_base_address`.

Create a `LoaderMockBackend` that implements `SyscallBackend` with VM support and can serve interpreter bytes when walked to the right path.

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-os -- elf_loader::tests`
Expected: FAIL — types don't exist

**Step 3: Implement**

1. Define auxv constants module: AT_NULL=0, AT_PHDR=3, AT_PHENT=4, AT_PHNUM=5, AT_PAGESZ=6, AT_BASE=7, AT_ENTRY=9, AT_RANDOM=25
2. Define `LoadResult { entry_point, auxv, exe_base, interp_base }` and `ElfLoadError` enum
3. Define `ElfLoader` trait with `fn load(&mut self, elf_bytes, backend) -> Result<LoadResult, ElfLoadError>`
4. Implement `InterpreterLoader`:
   - Parse executable ELF
   - Load PT_LOAD segments (ET_EXEC at vaddr, ET_DYN at configurable base)
   - If PT_INTERP: walk+read interpreter from backend, parse+load at non-overlapping base
   - Build auxv: AT_PHDR, AT_PHENT, AT_PHNUM, AT_PAGESZ, AT_ENTRY, AT_BASE (if interp), AT_RANDOM, AT_NULL
   - Return interpreter's entry (dynamic) or exe's entry (static)
5. Add `pub mod elf_loader;` to `lib.rs`

**Step 4: Run tests**

Run: `cargo test -p harmony-os -- elf_loader::tests`
Expected: PASS

**Step 5: Commit**

```
git add crates/harmony-os/src/elf_loader.rs crates/harmony-os/src/lib.rs
git commit -m "feat(elf-loader): add ElfLoader trait and InterpreterLoader"
```

---

### Task 7: Stack Layout Construction

Build the initial process stack with argc, argv, envp, auxv, and AT_RANDOM bytes.

**Files:**
- Modify: `crates/harmony-os/src/elf_loader.rs`

**Step 1: Write the failing tests**

Tests for: `build_stack_layout` (SP 16-byte aligned, argc correct), `stack_argv_readable` (argv[0] points to string), `stack_auxv_present` (auxv entries at correct offsets).

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-os -- elf_loader::tests`
Expected: FAIL — `build_initial_stack` doesn't exist

**Step 3: Implement**

`build_initial_stack(stack: &mut [u8], argv: &[&str], envp: &[&str], auxv: &[(u64, u64)], random_bytes: &[u8; 16]) -> u64`:
1. Write strings at top of stack (high addresses), track their addresses
2. Write AT_RANDOM 16 bytes, track address
3. Align to 16 bytes
4. Write structured part (growing down): argc, argv pointers, NULL, envp pointers, NULL, auxv pairs
5. Patch AT_RANDOM value in auxv to point to the random bytes
6. Return SP (16-byte aligned)

**Step 4: Run tests**

Run: `cargo test -p harmony-os -- elf_loader::tests`
Expected: PASS

**Step 5: Commit**

```
git add crates/harmony-os/src/elf_loader.rs
git commit -m "feat(elf-loader): add initial stack layout construction"
```

---

### Task 8: End-to-End Integration Tests

Connect InterpreterLoader + LibraryServer + stack setup. Verify with synthetic ELFs in an end-to-end test.

**Files:**
- Modify: `crates/harmony-os/src/elf_loader.rs`

**Step 1: Write integration tests**

`end_to_end_static_elf_load`: Build synthetic static ELF → InterpreterLoader → verify entry point, auxv, stack layout.

`end_to_end_dynamic_elf_with_interp`: Build synthetic ET_EXEC with PT_INTERP → provide interpreter bytes in mock backend → verify entry_point is interpreter's entry, auxv AT_ENTRY is exe's entry, AT_BASE is interpreter base.

**Step 2-4: Implement and verify**

The mock backend serves interpreter ELF bytes when the right path is walked. Uses `VmMockBackend`-style tracking for mapped regions.

**Step 5: Commit**

```
git add crates/harmony-os/src/elf_loader.rs
git commit -m "test(elf-loader): end-to-end integration tests"
```

---

### Task 9: Real Musl Integration Test

Test with pre-built musl binaries to verify real-world compatibility.

**Files:**
- Create: `crates/harmony-os/tests/fixtures/` (musl binaries)
- Create: `crates/harmony-os/tests/dynamic_elf.rs`

**Step 1: Prepare fixtures**

Cross-compile a hello world with `aarch64-linux-musl-gcc` and extract `ld-musl-aarch64.so.1` from the toolchain. Place in `tests/fixtures/`. If cross-compiler isn't available, mark tests as `#[ignore]`.

**Step 2: Write integration tests**

`parse_real_musl_hello`: `include_bytes!("fixtures/hello")` → `parse_elf` → verify interpreter contains "ld-musl", has segments.

`parse_real_ld_musl`: `include_bytes!("fixtures/ld-musl-aarch64.so.1")` → verify ET_DYN, no PT_INTERP, has segments.

**Step 3-4: Verify**

Run: `cargo test -p harmony-os --test dynamic_elf`
Expected: PASS (or IGNORED if no cross-compiler)

**Step 5: Commit**

```
git add crates/harmony-os/tests/
git commit -m "test: real musl integration test for dynamic ELF loading"
```

---

## Task Dependency Graph

```
Task 1 (ET_DYN) ──┐
                   ├── Task 6 (ElfLoader trait) ── Task 7 (Stack) ── Task 8 (Integration)
Task 2 (PT_INTERP)┘                                                        |
                                                                            v
Task 3 (LibraryServer) ──────────────────────────────────── Task 9 (Real musl test)
                                                                            ^
Task 4 (New syscalls) ───┐                                                  |
                         ├──────────────────────────────────────────────────┘
Task 5 (File mmap) ──────┘
```

Tasks 1-5 can be done in any order (independent). Tasks 6-7 depend on 1+2. Task 8 depends on 3+6+7. Task 9 depends on everything.

## Quality Gates

Before delivering:

```
cargo test --workspace
cargo clippy --workspace
cargo fmt --all -- --check
```

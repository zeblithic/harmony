# Dynamic ELF Loading Design (harmony-ms6)

## Goal

Enable dynamically-linked aarch64 musl binaries to run on Harmony. MVP: a "hello world" program compiled with `aarch64-linux-musl-gcc` that prints via printf and exits.

## Architecture

The kernel handles the minimum necessary for dynamic loading — parse PT_INTERP, load the interpreter (ld-musl) alongside the executable, construct the auxiliary vector, set up the initial stack, and jump to the interpreter's entry point. ld-musl (unmodified) handles all dynamic linking (relocations, symbol resolution, .so loading) via standard Linux syscalls.

Libraries are served from a virtual `/lib` 9P namespace backed by a name-to-blob manifest. When ld-musl calls `open("/lib/libc.so")`, the namespace resolves the name to raw ELF bytes. This evolves to CID-based ContentServer lookups when the Zenoh content store is ready.

### Loading Flow

```
kernel: parse_elf(executable)
  → PT_INTERP found: "/lib/ld-musl-aarch64.so.1"
kernel: open("/lib/ld-musl-aarch64.so.1") via LibraryServer
kernel: parse_elf(interpreter)
kernel: mmap executable segments at vaddr (ET_EXEC) or base+vaddr (ET_DYN)
kernel: mmap interpreter segments at non-overlapping base (e.g., 0x7f000000)
kernel: build auxv (AT_PHDR, AT_ENTRY, AT_BASE, AT_PHNUM, ...)
kernel: build stack (argc, argv, envp, auxv, AT_RANDOM bytes)
kernel: jump to interpreter entry point
  ld-musl: open("/lib/libc.so") → mmap → relocate → resolve symbols
  ld-musl: jump to executable entry point
  executable: printf("Hello") → write syscall → Linuxulator → serial
  executable: exit(0) → exit_group syscall → Linuxulator → halt
```

## Components

### 1. ELF Parser Extensions (`elf.rs`)

Extend the existing `parse_elf()` to support dynamic binaries:

- **Accept ET_DYN** alongside ET_EXEC. Shared libraries and PIE executables are ET_DYN. The caller decides the base address for ET_DYN files.
- **Parse PT_INTERP** — extract interpreter path string. Add `interpreter: Option<String>` to `ParsedElf`.
- **Expose phdr metadata** — add `phdr_vaddr`, `phdr_entry_size`, `phdr_count` to `ParsedElf` for auxv construction (AT_PHDR, AT_PHENT, AT_PHNUM).
- **New error variant** — `InterpreterPathInvalid` for malformed PT_INTERP segments.
- **No section headers** — PT_DYNAMIC, PT_GNU_RELRO, PT_NOTE etc. are ignored. ld-musl reads these from mapped memory.

### 2. ElfLoader Trait (`elf_loader.rs`)

Abstraction over loading strategy, designed for future extensibility:

```rust
pub trait ElfLoader {
    fn load(
        &mut self,
        elf_bytes: &[u8],
        backend: &mut dyn SyscallBackend,
    ) -> Result<LoadResult, ElfLoadError>;
}

pub struct LoadResult {
    pub entry_point: u64,       // interpreter's entry (dynamic) or exe's entry (static)
    pub auxv: Vec<(u64, u64)>,  // auxiliary vector entries
    pub exe_base: u64,          // base address for ET_DYN, 0 for ET_EXEC
}
```

**InterpreterLoader** (first implementation):

1. Parse executable ELF
2. Load segments: ET_EXEC at specified vaddrs, ET_DYN at chosen base
3. If PT_INTERP: read interpreter from `/lib` via backend, parse and load at non-overlapping base (e.g., 0x7f000000)
4. Build auxv with AT_PHDR, AT_PHENT, AT_PHNUM, AT_ENTRY, AT_BASE, AT_PAGESZ, AT_RANDOM, AT_NULL
5. Return LoadResult with interpreter's entry point

**Static fallback:** No PT_INTERP → return executable's entry point with minimal auxv. The trait unifies both paths.

### 3. File-Backed mmap (`linuxulator.rs`)

When `mmap` is called with a valid fd (not MAP_ANONYMOUS):

1. Look up fd → 9P fid in fd table
2. Read `length` bytes from file at `offset` via `backend.read(fid, offset, count)`
3. Allocate pages via `backend.vm_mmap()` (or arena fallback)
4. Copy file content into mapped pages
5. Apply permissions via `vm_mprotect()`
6. Zero-fill any remainder (freshly allocated pages are already zeroed)

This is eager-read — no demand paging. Libraries are small (~700KB for musl libc), and the abort handler currently panics rather than handling page faults.

### 4. LibraryServer (`library_server.rs`)

A flat, read-only 9P FileServer mapping library names to raw ELF bytes:

```rust
pub struct LibraryServer {
    manifest: BTreeMap<String, Vec<u8>>,  // "ld-musl-aarch64.so.1" → ELF bytes
    // ... fid tracking
}

impl FileServer for LibraryServer { ... }
```

- `walk("ld-musl-aarch64.so.1")` → look up name in manifest
- `open(fid, OREAD)` → mark open
- `read(fid, offset, count)` → return slice of blob bytes
- `stat(fid)` → return file size
- `clunk(fid)` → release fid

No directory listing, no nested paths, no write support. Minimal surface.

**Evolution:** When the Zenoh content store is ready, the manifest lookup becomes a CID resolution. Libraries are "00" (public durable) content — automatically shared and replicated. The FileServer interface stays the same; only the backing store changes.

### 5. Stack Layout and Auxiliary Vector

Initial stack built by the kernel before jumping to the interpreter:

```
High addresses
┌─────────────────────┐
│ AT_RANDOM 16 bytes  │  ← random bytes for stack canary
├─────────────────────┤
│ argv[0] string      │  ← "./hello\0"
├─────────────────────┤
│ padding (16-byte)   │
├─────────────────────┤
│ AT_NULL (0, 0)      │
│ AT_RANDOM (25, ptr) │
│ AT_PAGESZ (6, 4096) │
│ AT_BASE (7, interp) │  ← interpreter load address
│ AT_ENTRY (9, exe)   │  ← executable's entry point
│ AT_PHNUM (5, n)     │
│ AT_PHENT (4, 56)    │
│ AT_PHDR (3, addr)   │  ← executable's phdrs in memory
├─────────────────────┤
│ NULL (envp term)    │
├─────────────────────┤
│ NULL (argv term)    │
│ argv[0] pointer     │
├─────────────────────┤
│ argc = 1            │  ← SP points here
└─────────────────────┘
Low addresses
```

- Stack allocated via mmap at a high address (e.g., 0x7FFE0000, 64KB)
- AT_RANDOM points to 16 random bytes on the stack
- AT_BASE = 0 if no interpreter
- argc/argv minimal — just binary name; real args are a straightforward extension later

### 6. New Syscalls

Syscalls ld-musl needs beyond what exists today:

| Syscall | Purpose | Implementation |
|---------|---------|----------------|
| `writev` | Scatter-gather write (musl printf uses this) | Concatenate iovecs, delegate to existing write path |
| `lseek` | Seek within .so files before mmap | Track per-fd offset, return new position |
| `getrandom` | Stack canary, AT_RANDOM seeding | Fill buffer from hardware RNG (RNDR on aarch64) |
| `getcwd` | Path resolution during library search | Return "/" |
| `readlink` | `/proc/self/exe` resolution | Return ENOSYS (ld-musl has fallbacks) |

Existing syscalls extended:
- `mmap` — add file-backed support (currently anonymous-only)
- `openat` — must work for `/lib/` paths through namespace

## File Organization

| Component | Location | New/Modified |
|-----------|----------|-------------|
| ELF parser extensions | `crates/harmony-os/src/elf.rs` | Modified |
| ElfLoader trait + InterpreterLoader | `crates/harmony-os/src/elf_loader.rs` | New |
| File-backed mmap, new syscalls | `crates/harmony-os/src/linuxulator.rs` | Modified |
| LinuxSyscall enum extensions | `crates/harmony-os/src/linuxulator.rs` | Modified |
| LibraryServer | `crates/harmony-microkernel/src/library_server.rs` | New |
| Test musl binaries | `crates/harmony-os/tests/fixtures/` | New |
| Integration test | `crates/harmony-os/tests/dynamic_elf.rs` | New |

Not touched: `harmony-boot-aarch64` (boot stub doesn't load ELF yet — separate bead), `harmony-unikernel`, `harmony-crypto`, `harmony-identity`.

## Testing Strategy

**Unit tests (synthetic ELFs):**
- PT_INTERP parsing — verify `parsed.interpreter` field
- ET_DYN acceptance — verify parser accepts, base address offset works
- InterpreterLoader — mock backend, verify both ELFs loaded, correct auxv, correct entry point
- Static fallback — no PT_INTERP returns exe entry directly
- Auxv construction — verify AT_PHDR, AT_ENTRY, AT_BASE values
- Stack layout — verify argc/argv/envp/auxv at correct SP offsets
- File-backed mmap — mmap with fd, verify file content in mapped pages
- LibraryServer — walk/open/read/stat against manifest

**Integration tests (real musl binaries):**
- Cross-compile hello.c with `aarch64-linux-musl-gcc`
- Load through InterpreterLoader with real ld-musl and libc.so in LibraryServer
- Verify Linuxulator processes write syscall with "Hello" output and exit with code 0
- Runs on host via Linuxulator + mock backends (not QEMU)

## What We Skip (YAGNI)

- No section headers, relocations, or symbol tables in kernel (ld-musl handles these)
- No demand paging (eager-read for file-backed mmap)
- No directory listing in LibraryServer
- No real argc/argv beyond binary name
- No AT_UID/AT_GID (no users), AT_HWCAP (stub as 0)
- No execve syscall (separate concern — wiring loader into boot/process lifecycle)

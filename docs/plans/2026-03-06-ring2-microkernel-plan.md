# Ring 2 Microkernel Milestone A — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Two processes communicating over 9P-inspired IPC with UCAN capability enforcement in the harmony-os microkernel.

**Architecture:** Ring 2 layers on Ring 1. The microkernel manages a process table where each process implements a `FileServer` trait. Cross-process IPC is dispatched through a `Kernel` struct that resolves per-process namespaces and checks UCAN capability tokens before calling the target's FileServer methods. All sans-I/O — testable on the host.

**Tech Stack:** Rust (no_std compatible with alloc), harmony-identity (UCAN tokens), harmony-platform (EntropySource trait), harmony-unikernel (Ring 1 runtime)

**Design doc:** `docs/plans/2026-03-06-ring2-microkernel-design.md`

---

## Codebase Orientation

### Repo: `zeblithic/harmony-os`

```
crates/
  harmony-microkernel/     ← WE ARE BUILDING HERE
    Cargo.toml             ← Modify (add test-utils feature)
    src/
      lib.rs               ← Modify (core types, trait, module declarations)
      echo.rs              ← Create (EchoServer FileServer impl)
      namespace.rs         ← Create (per-process mount table)
      kernel.rs            ← Create (Kernel struct, IPC dispatch)
      serial_server.rs     ← Create (serial output as FileServer)
  harmony-unikernel/       ← Ring 1 (DO NOT MODIFY)
  harmony-boot/            ← Ring 1 boot (modify for ring2 feature in Task 8)
  harmony-os/              ← Ring 3 scaffold (DO NOT MODIFY)
```

### Key imports you'll need

```rust
// From harmony-identity (Ring 0)
use harmony_identity::{
    PrivateIdentity, Identity,
    UcanToken, CapabilityType, UcanError,
    verify_token, ProofResolver, IdentityResolver, RevocationSet,
    MemoryProofStore, MemoryIdentityStore, MemoryRevocationSet,  // behind test-utils feature
};

// From harmony-platform (Ring 0)
use harmony_platform::EntropySource;

// From harmony-unikernel (Ring 1)
use harmony_unikernel::{KernelEntropy, MemoryState, UnikernelRuntime};
```

### Test commands

```bash
cargo test -p harmony-microkernel            # Run all microkernel tests
cargo test -p harmony-microkernel test_name   # Run single test
cargo clippy --workspace                      # Lint
cargo test --workspace                        # All workspace tests
```

### Test entropy helper (reuse across tasks)

```rust
fn make_test_entropy() -> KernelEntropy<impl FnMut(&mut [u8])> {
    let mut seed = 42u64;
    KernelEntropy::new(move |buf: &mut [u8]| {
        for b in buf.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (seed >> 33) as u8;
        }
    })
}
```

---

## Task 1: Core Types and FileServer Trait

**Files:**
- Modify: `crates/harmony-microkernel/Cargo.toml`
- Modify: `crates/harmony-microkernel/src/lib.rs`

This task sets up the foundational types and the `FileServer` trait that every Ring 2 process implements.

**Step 1: Update Cargo.toml**

Replace the contents of `crates/harmony-microkernel/Cargo.toml` with:

```toml
[package]
name = "harmony-microkernel"
description = "Ring 2: Microkernel with 9P IPC, capability enforcement, process isolation"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
# Ring 1
harmony-unikernel = { workspace = true, features = ["std"] }

# Ring 0 — protocol crates needed at the kernel level
harmony-crypto = { workspace = true, features = ["std"] }
harmony-identity = { workspace = true, features = ["std", "test-utils"] }
harmony-platform = { workspace = true, features = ["std"] }

[dev-dependencies]
```

Note: `test-utils` on harmony-identity gives us `MemoryProofStore`, `MemoryIdentityStore`, `MemoryRevocationSet` — the UCAN verification stores the kernel needs.

**Step 2: Write lib.rs with core types and a compile test**

Replace the contents of `crates/harmony-microkernel/src/lib.rs` with:

```rust
// SPDX-License-Identifier: GPL-2.0-or-later

//! # Harmony Microkernel (Ring 2)
//!
//! Adds three capabilities to the unikernel foundation:
//! - **Process isolation** — cooperative, trait-object-based (hardware paging is future work)
//! - **9P-inspired IPC** — every process implements `FileServer`
//! - **Capability enforcement** — UCAN tokens gate all cross-process IPC

extern crate alloc;

pub mod echo;
pub mod kernel;
pub mod namespace;
pub mod serial_server;

use alloc::sync::Arc;
use alloc::vec::Vec;

// ── Fundamental identifiers ──────────────────────────────────────────

/// File identifier — a per-session handle to an open or walked file.
pub type Fid = u32;

/// Unique file identity — like an inode number. Stable across opens.
pub type QPath = u64;

// ── Enums ────────────────────────────────────────────────────────────

/// How a file is opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenMode {
    Read,
    Write,
    ReadWrite,
}

/// What kind of entry a file is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Regular,
    Directory,
}

/// Errors returned by IPC operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpcError {
    NotFound,
    PermissionDenied,
    NotOpen,
    InvalidFid,
    NotDirectory,
    ReadOnly,
}

// ── File metadata ────────────────────────────────────────────────────

/// Metadata about a file (like 9P's stat).
#[derive(Debug, Clone)]
pub struct FileStat {
    pub qpath: QPath,
    pub name: Arc<str>,
    pub size: u64,
    pub file_type: FileType,
}

// ── FileServer trait ─────────────────────────────────────────────────

/// The heart of Ring 2: every process implements this trait.
///
/// Mirrors 9P2000 semantics (walk, open, read, write, clunk, stat)
/// but uses Rust types instead of wire-format bytes.
pub trait FileServer {
    /// Walk from `fid` to a child named `name`, assigning `new_fid`.
    /// Returns the new file's QPath.
    fn walk(&mut self, fid: Fid, new_fid: Fid, name: &str) -> Result<QPath, IpcError>;

    /// Open `fid` with the given mode.
    fn open(&mut self, fid: Fid, mode: OpenMode) -> Result<(), IpcError>;

    /// Read up to `count` bytes at `offset` from an open fid.
    fn read(&mut self, fid: Fid, offset: u64, count: u32) -> Result<Vec<u8>, IpcError>;

    /// Write `data` at `offset` to an open fid. Returns bytes written.
    fn write(&mut self, fid: Fid, offset: u64, data: &[u8]) -> Result<u32, IpcError>;

    /// Release a fid (like 9P's clunk).
    fn clunk(&mut self, fid: Fid) -> Result<(), IpcError>;

    /// Stat a fid — returns name, size, type.
    fn stat(&mut self, fid: Fid) -> Result<FileStat, IpcError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that FileServer is object-safe (can be used as Box<dyn FileServer>).
    struct NullServer;

    impl FileServer for NullServer {
        fn walk(&mut self, _: Fid, _: Fid, _: &str) -> Result<QPath, IpcError> {
            Err(IpcError::NotFound)
        }
        fn open(&mut self, _: Fid, _: OpenMode) -> Result<(), IpcError> {
            Err(IpcError::InvalidFid)
        }
        fn read(&mut self, _: Fid, _: u64, _: u32) -> Result<Vec<u8>, IpcError> {
            Err(IpcError::InvalidFid)
        }
        fn write(&mut self, _: Fid, _: u64, _: &[u8]) -> Result<u32, IpcError> {
            Err(IpcError::InvalidFid)
        }
        fn clunk(&mut self, _: Fid) -> Result<(), IpcError> {
            Err(IpcError::InvalidFid)
        }
        fn stat(&mut self, _: Fid) -> Result<FileStat, IpcError> {
            Err(IpcError::InvalidFid)
        }
    }

    #[test]
    fn file_server_is_object_safe() {
        let server: Box<dyn FileServer> = Box::new(NullServer);
        assert!(server.stat(0).is_err());
    }
}
```

**Step 3: Create empty module files**

Create placeholder files so the module declarations compile:

`crates/harmony-microkernel/src/echo.rs`:
```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! EchoServer — a trivial FileServer for testing IPC.
```

`crates/harmony-microkernel/src/namespace.rs`:
```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! Per-process namespace (mount table + path resolution).
```

`crates/harmony-microkernel/src/kernel.rs`:
```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! Microkernel — process table, IPC dispatch, capability enforcement.
```

`crates/harmony-microkernel/src/serial_server.rs`:
```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! KernelSerialServer — serial output as a FileServer.
```

**Step 4: Run tests**

```bash
cargo test -p harmony-microkernel
```

Expected: 1 test passes (`file_server_is_object_safe`).

**Step 5: Lint**

```bash
cargo clippy --workspace
```

Expected: zero warnings.

**Step 6: Commit**

```bash
git add crates/harmony-microkernel/
git commit -m "feat(microkernel): core types and FileServer trait (Ring 2 foundation)"
```

---

## Task 2: EchoServer

**Files:**
- Modify: `crates/harmony-microkernel/src/echo.rs`

The EchoServer is the first FileServer implementation. It proves the trait
works and gives us something to test IPC against in later tasks. It has a
root directory containing two files:

- `hello` — read-only, returns `"Hello from echo server!"`
- `echo` — writable, returns whatever was last written

**Step 1: Write failing tests**

Add to `crates/harmony-microkernel/src/echo.rs`:

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! EchoServer — a trivial FileServer for testing IPC.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::{Fid, FileServer, FileStat, FileType, IpcError, OpenMode, QPath};

// Implementation goes here (Step 3)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walk_hello() {
        let mut srv = EchoServer::new();
        // fid 0 = root, walk to "hello" assigning fid 1
        let qpath = srv.walk(0, 1, "hello").unwrap();
        assert_eq!(qpath, 1); // hello has qpath 1
    }

    #[test]
    fn walk_echo() {
        let mut srv = EchoServer::new();
        let qpath = srv.walk(0, 1, "echo").unwrap();
        assert_eq!(qpath, 2); // echo has qpath 2
    }

    #[test]
    fn walk_invalid_name() {
        let mut srv = EchoServer::new();
        assert_eq!(srv.walk(0, 1, "nonexistent"), Err(IpcError::NotFound));
    }

    #[test]
    fn walk_from_invalid_fid() {
        let mut srv = EchoServer::new();
        assert_eq!(srv.walk(99, 1, "hello"), Err(IpcError::InvalidFid));
    }

    #[test]
    fn read_hello() {
        let mut srv = EchoServer::new();
        srv.walk(0, 1, "hello").unwrap();
        srv.open(1, OpenMode::Read).unwrap();
        let data = srv.read(1, 0, 256).unwrap();
        assert_eq!(data, b"Hello from echo server!");
    }

    #[test]
    fn read_without_open() {
        let mut srv = EchoServer::new();
        srv.walk(0, 1, "hello").unwrap();
        assert_eq!(srv.read(1, 0, 256), Err(IpcError::NotOpen));
    }

    #[test]
    fn write_then_read_echo() {
        let mut srv = EchoServer::new();
        srv.walk(0, 1, "echo").unwrap();
        srv.open(1, OpenMode::ReadWrite).unwrap();
        let written = srv.write(1, 0, b"test data").unwrap();
        assert_eq!(written, 9);
        let data = srv.read(1, 0, 256).unwrap();
        assert_eq!(data, b"test data");
    }

    #[test]
    fn write_hello_is_read_only() {
        let mut srv = EchoServer::new();
        srv.walk(0, 1, "hello").unwrap();
        srv.open(1, OpenMode::Write).unwrap();
        assert_eq!(srv.write(1, 0, b"nope"), Err(IpcError::ReadOnly));
    }

    #[test]
    fn clunk_releases_fid() {
        let mut srv = EchoServer::new();
        srv.walk(0, 1, "hello").unwrap();
        srv.clunk(1).unwrap();
        // fid 1 no longer valid
        assert_eq!(srv.open(1, OpenMode::Read), Err(IpcError::InvalidFid));
    }

    #[test]
    fn stat_root() {
        let mut srv = EchoServer::new();
        let st = srv.stat(0).unwrap();
        assert_eq!(st.file_type, FileType::Directory);
        assert_eq!(&*st.name, "/");
    }

    #[test]
    fn stat_hello() {
        let mut srv = EchoServer::new();
        srv.walk(0, 1, "hello").unwrap();
        let st = srv.stat(1).unwrap();
        assert_eq!(st.file_type, FileType::Regular);
        assert_eq!(&*st.name, "hello");
    }
}
```

**Step 2: Run tests to verify they fail**

```bash
cargo test -p harmony-microkernel
```

Expected: FAIL — `EchoServer` not found.

**Step 3: Implement EchoServer**

Add the implementation above the `#[cfg(test)]` section in `echo.rs`:

```rust
/// QPath constants for the echo server's files.
const QPATH_ROOT: QPath = 0;
const QPATH_HELLO: QPath = 1;
const QPATH_ECHO: QPath = 2;

/// Per-fid state tracked by the echo server.
struct FidState {
    qpath: QPath,
    is_open: bool,
    mode: Option<OpenMode>,
}

/// A trivial FileServer with two files: `hello` (read-only greeting)
/// and `echo` (write bytes in, read them back).
pub struct EchoServer {
    fids: BTreeMap<Fid, FidState>,
    echo_data: Vec<u8>,
}

impl EchoServer {
    pub fn new() -> Self {
        let mut fids = BTreeMap::new();
        // fid 0 = root directory (always present)
        fids.insert(0, FidState { qpath: QPATH_ROOT, is_open: false, mode: None });
        EchoServer { fids, echo_data: Vec::new() }
    }
}

impl FileServer for EchoServer {
    fn walk(&mut self, fid: Fid, new_fid: Fid, name: &str) -> Result<QPath, IpcError> {
        let state = self.fids.get(&fid).ok_or(IpcError::InvalidFid)?;
        if state.qpath != QPATH_ROOT {
            return Err(IpcError::NotDirectory);
        }
        let qpath = match name {
            "hello" => QPATH_HELLO,
            "echo" => QPATH_ECHO,
            _ => return Err(IpcError::NotFound),
        };
        self.fids.insert(new_fid, FidState { qpath, is_open: false, mode: None });
        Ok(qpath)
    }

    fn open(&mut self, fid: Fid, mode: OpenMode) -> Result<(), IpcError> {
        let state = self.fids.get_mut(&fid).ok_or(IpcError::InvalidFid)?;
        state.is_open = true;
        state.mode = Some(mode);
        Ok(())
    }

    fn read(&mut self, fid: Fid, _offset: u64, count: u32) -> Result<Vec<u8>, IpcError> {
        let state = self.fids.get(&fid).ok_or(IpcError::InvalidFid)?;
        if !state.is_open {
            return Err(IpcError::NotOpen);
        }
        let data = match state.qpath {
            QPATH_HELLO => b"Hello from echo server!".to_vec(),
            QPATH_ECHO => {
                let end = core::cmp::min(self.echo_data.len(), count as usize);
                self.echo_data[..end].to_vec()
            }
            _ => return Err(IpcError::NotFound),
        };
        Ok(data)
    }

    fn write(&mut self, fid: Fid, _offset: u64, data: &[u8]) -> Result<u32, IpcError> {
        let state = self.fids.get(&fid).ok_or(IpcError::InvalidFid)?;
        if !state.is_open {
            return Err(IpcError::NotOpen);
        }
        match state.qpath {
            QPATH_HELLO => Err(IpcError::ReadOnly),
            QPATH_ECHO => {
                self.echo_data = data.to_vec();
                Ok(data.len() as u32)
            }
            _ => Err(IpcError::NotFound),
        }
    }

    fn clunk(&mut self, fid: Fid) -> Result<(), IpcError> {
        self.fids.remove(&fid).ok_or(IpcError::InvalidFid)?;
        Ok(())
    }

    fn stat(&mut self, fid: Fid) -> Result<FileStat, IpcError> {
        let state = self.fids.get(&fid).ok_or(IpcError::InvalidFid)?;
        let (name, file_type, size) = match state.qpath {
            QPATH_ROOT => ("/", FileType::Directory, 0),
            QPATH_HELLO => ("hello", FileType::Regular, 23),
            QPATH_ECHO => ("echo", FileType::Regular, self.echo_data.len() as u64),
            _ => return Err(IpcError::NotFound),
        };
        Ok(FileStat {
            qpath: state.qpath,
            name: Arc::from(name),
            size,
            file_type,
        })
    }
}
```

**Step 4: Run tests to verify they pass**

```bash
cargo test -p harmony-microkernel
```

Expected: 11 tests pass (1 from Task 1 + 10 new).

**Step 5: Lint**

```bash
cargo clippy --workspace
```

**Step 6: Commit**

```bash
git add crates/harmony-microkernel/src/echo.rs
git commit -m "feat(microkernel): EchoServer — first FileServer implementation"
```

---

## Task 3: Namespace

**Files:**
- Modify: `crates/harmony-microkernel/src/namespace.rs`

The namespace is a per-process mount table. It maps path prefixes to
target process IDs. `resolve()` finds the longest matching prefix and
returns the target PID plus the remaining path after the prefix.

**Step 1: Write failing tests**

Add to `crates/harmony-microkernel/src/namespace.rs`:

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! Per-process namespace (mount table + path resolution).

use alloc::collections::BTreeMap;
use alloc::sync::Arc;

use crate::Fid;

// Implementation goes here (Step 3)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_simple_mount() {
        let mut ns = Namespace::new();
        ns.mount("/echo", 1, 0);
        let (mp, remainder) = ns.resolve("/echo/hello").unwrap();
        assert_eq!(mp.target_pid, 1);
        assert_eq!(remainder, "hello");
    }

    #[test]
    fn resolve_mount_root() {
        let mut ns = Namespace::new();
        ns.mount("/echo", 1, 0);
        // Accessing the mount point itself — remainder is empty
        let (mp, remainder) = ns.resolve("/echo").unwrap();
        assert_eq!(mp.target_pid, 1);
        assert_eq!(remainder, "");
    }

    #[test]
    fn resolve_nested_path() {
        let mut ns = Namespace::new();
        ns.mount("/dev/serial", 0, 0);
        let (mp, remainder) = ns.resolve("/dev/serial").unwrap();
        assert_eq!(mp.target_pid, 0);
        assert_eq!(remainder, "");
    }

    #[test]
    fn resolve_longest_prefix_match() {
        let mut ns = Namespace::new();
        ns.mount("/a", 1, 0);
        ns.mount("/a/b", 2, 0);
        // "/a/b/c" should match "/a/b" (longer), not "/a"
        let (mp, remainder) = ns.resolve("/a/b/c").unwrap();
        assert_eq!(mp.target_pid, 2);
        assert_eq!(remainder, "c");
    }

    #[test]
    fn resolve_unmounted_path() {
        let ns = Namespace::new();
        assert!(ns.resolve("/nonexistent").is_none());
    }

    #[test]
    fn resolve_partial_prefix_no_match() {
        let mut ns = Namespace::new();
        ns.mount("/echo", 1, 0);
        // "/echooo" should NOT match "/echo" — must be exact prefix + "/" boundary
        assert!(ns.resolve("/echooo").is_none());
    }

    #[test]
    fn mount_preserves_root_fid() {
        let mut ns = Namespace::new();
        ns.mount("/data", 3, 42);
        let (mp, _) = ns.resolve("/data/file").unwrap();
        assert_eq!(mp.root_fid, 42);
    }
}
```

**Step 2: Run tests to verify they fail**

```bash
cargo test -p harmony-microkernel
```

Expected: FAIL — `Namespace` not found.

**Step 3: Implement Namespace**

Add the implementation above the `#[cfg(test)]` section:

```rust
/// A mount point maps a path prefix to a target process.
#[derive(Debug, Clone)]
pub struct MountPoint {
    pub target_pid: u32,
    pub root_fid: Fid,
}

/// Per-process namespace — a mount table mapping path prefixes to servers.
pub struct Namespace {
    mounts: BTreeMap<Arc<str>, MountPoint>,
}

impl Namespace {
    pub fn new() -> Self {
        Namespace { mounts: BTreeMap::new() }
    }

    /// Mount a server at `path`. Subsequent resolves matching this prefix
    /// will route to `target_pid`.
    pub fn mount(&mut self, path: &str, target_pid: u32, root_fid: Fid) {
        self.mounts.insert(Arc::from(path), MountPoint { target_pid, root_fid });
    }

    /// Resolve `path` to a mount point and remainder.
    ///
    /// Finds the longest matching prefix. Returns `None` if no mount matches.
    /// The prefix must match at a "/" boundary or exactly (no partial matches).
    pub fn resolve(&self, path: &str) -> Option<(&MountPoint, &str)> {
        let mut best: Option<(&MountPoint, &str)> = None;
        let mut best_len = 0;

        for (prefix, mount) in &self.mounts {
            let prefix_str: &str = prefix;
            if path == prefix_str {
                // Exact match — remainder is empty
                if prefix_str.len() > best_len {
                    best = Some((mount, ""));
                    best_len = prefix_str.len();
                }
            } else if path.starts_with(prefix_str) {
                // Check for "/" boundary after prefix to avoid partial matches
                let after = &path[prefix_str.len()..];
                if after.starts_with('/') {
                    let remainder = &after[1..]; // strip the leading "/"
                    if prefix_str.len() > best_len {
                        best = Some((mount, remainder));
                        best_len = prefix_str.len();
                    }
                }
            }
        }

        best
    }
}
```

**Step 4: Run tests to verify they pass**

```bash
cargo test -p harmony-microkernel
```

Expected: 18 tests pass (11 prior + 7 new).

**Step 5: Lint**

```bash
cargo clippy --workspace
```

**Step 6: Commit**

```bash
git add crates/harmony-microkernel/src/namespace.rs
git commit -m "feat(microkernel): Namespace — per-process mount table with path resolution"
```

---

## Task 4: Kernel Struct and Capability Enforcement

**Files:**
- Modify: `crates/harmony-microkernel/src/kernel.rs`

The Kernel manages the process table, capability verification, and IPC
dispatch. This task builds the struct, process spawning, and capability
checking. Task 5 adds IPC dispatch on top.

**Step 1: Write failing tests**

Add to `crates/harmony-microkernel/src/kernel.rs`:

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! Microkernel — process table, IPC dispatch, capability enforcement.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;

use harmony_identity::{
    CapabilityType, IdentityResolver, MemoryIdentityStore, MemoryProofStore,
    MemoryRevocationSet, PrivateIdentity, ProofResolver, RevocationSet, UcanToken,
    verify_token,
};
use harmony_platform::EntropySource;

use crate::echo::EchoServer;
use crate::namespace::{MountPoint, Namespace};
use crate::{Fid, FileServer, IpcError, OpenMode, QPath};

// Implementation goes here (Step 3)

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_unikernel::KernelEntropy;

    fn make_test_entropy() -> KernelEntropy<impl FnMut(&mut [u8])> {
        let mut seed = 42u64;
        KernelEntropy::new(move |buf: &mut [u8]| {
            for b in buf.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                *b = (seed >> 33) as u8;
            }
        })
    }

    #[test]
    fn spawn_process_assigns_pid() {
        let mut entropy = make_test_entropy();
        let kernel_id = PrivateIdentity::generate(&mut entropy);
        let mut kernel = Kernel::new(kernel_id);

        let pid = kernel.spawn_process(
            "echo",
            Box::new(EchoServer::new()),
            &[],
        );
        assert_eq!(pid, 0);

        let pid2 = kernel.spawn_process(
            "echo2",
            Box::new(EchoServer::new()),
            &[],
        );
        assert_eq!(pid2, 1);
    }

    #[test]
    fn capability_check_with_valid_cap() {
        let mut entropy = make_test_entropy();
        let kernel_id = PrivateIdentity::generate(&mut entropy);
        let mut kernel = Kernel::new(kernel_id.clone());

        let process_addr = [0x01u8; 16];
        let cap = kernel_id.issue_root_token(
            &mut entropy,
            &process_addr,
            CapabilityType::Endpoint,
            b"pid:1",
            0, 0,
        ).unwrap();

        kernel.identity_store.insert(kernel_id.public_identity().clone());

        assert!(kernel.check_endpoint_cap(&[cap], &process_addr, 1, 0).is_ok());
    }

    #[test]
    fn capability_check_no_cap() {
        let mut entropy = make_test_entropy();
        let kernel_id = PrivateIdentity::generate(&mut entropy);
        let kernel = Kernel::new(kernel_id);

        let process_addr = [0x01u8; 16];
        assert_eq!(
            kernel.check_endpoint_cap(&[], &process_addr, 1, 0),
            Err(IpcError::PermissionDenied)
        );
    }

    #[test]
    fn capability_check_wrong_pid() {
        let mut entropy = make_test_entropy();
        let kernel_id = PrivateIdentity::generate(&mut entropy);
        let mut kernel = Kernel::new(kernel_id.clone());

        let process_addr = [0x01u8; 16];
        let cap = kernel_id.issue_root_token(
            &mut entropy,
            &process_addr,
            CapabilityType::Endpoint,
            b"pid:1",
            0, 0,
        ).unwrap();

        kernel.identity_store.insert(kernel_id.public_identity().clone());

        // Cap is for pid:1, trying to access pid:2
        assert_eq!(
            kernel.check_endpoint_cap(&[cap], &process_addr, 2, 0),
            Err(IpcError::PermissionDenied)
        );
    }

    #[test]
    fn capability_check_wildcard() {
        let mut entropy = make_test_entropy();
        let kernel_id = PrivateIdentity::generate(&mut entropy);
        let mut kernel = Kernel::new(kernel_id.clone());

        let process_addr = [0x01u8; 16];
        let cap = kernel_id.issue_root_token(
            &mut entropy,
            &process_addr,
            CapabilityType::Endpoint,
            b"*",
            0, 0,
        ).unwrap();

        kernel.identity_store.insert(kernel_id.public_identity().clone());

        // Wildcard should match any pid
        assert!(kernel.check_endpoint_cap(&[cap], &process_addr, 1, 0).is_ok());
        assert!(kernel.check_endpoint_cap(&[cap], &process_addr, 99, 0).is_ok());
    }
}
```

**Step 2: Run tests to verify they fail**

```bash
cargo test -p harmony-microkernel
```

Expected: FAIL — `Kernel` not found.

**Step 3: Implement Kernel struct and capability checking**

Add the implementation above the `#[cfg(test)]` section:

```rust
/// A process in the microkernel.
pub struct Process {
    pub pid: u32,
    pub name: Arc<str>,
    pub namespace: Namespace,
    pub capabilities: Vec<UcanToken>,
    pub address_hash: [u8; 16],
    pub(crate) server: Box<dyn FileServer>,
}

/// The microkernel: process table, IPC dispatch, capability enforcement.
pub struct Kernel {
    processes: BTreeMap<u32, Process>,
    next_pid: u32,
    identity: PrivateIdentity,
    pub(crate) identity_store: MemoryIdentityStore,
    proof_store: MemoryProofStore,
    revocations: MemoryRevocationSet,
    /// Maps (client_pid, client_fid) → target_pid for open fids.
    fid_owners: BTreeMap<(u32, Fid), u32>,
}

impl Kernel {
    /// Create a new microkernel with the given identity.
    pub fn new(identity: PrivateIdentity) -> Self {
        let mut identity_store = MemoryIdentityStore::new();
        identity_store.insert(identity.public_identity().clone());
        Kernel {
            processes: BTreeMap::new(),
            next_pid: 0,
            identity,
            identity_store,
            proof_store: MemoryProofStore::new(),
            revocations: MemoryRevocationSet::new(),
            fid_owners: BTreeMap::new(),
        }
    }

    /// Spawn a process. Returns the assigned PID.
    ///
    /// `mounts` are (path, target_pid, root_fid) tuples to pre-populate
    /// the process's namespace.
    pub fn spawn_process(
        &mut self,
        name: &str,
        server: Box<dyn FileServer>,
        mounts: &[(&str, u32, Fid)],
    ) -> u32 {
        let pid = self.next_pid;
        self.next_pid += 1;

        // Derive a simple address hash from the pid.
        let mut address_hash = [0u8; 16];
        address_hash[..4].copy_from_slice(&pid.to_be_bytes());

        let mut namespace = Namespace::new();
        for &(path, target_pid, root_fid) in mounts {
            namespace.mount(path, target_pid, root_fid);
        }

        self.processes.insert(pid, Process {
            pid,
            name: Arc::from(name),
            namespace,
            capabilities: Vec::new(),
            address_hash,
            server,
        });

        pid
    }

    /// Grant an endpoint capability to a process, allowing it to access
    /// the target process's FileServer via IPC.
    pub fn grant_endpoint_cap(
        &mut self,
        entropy: &mut impl EntropySource,
        process_pid: u32,
        target_pid: u32,
        now: u64,
    ) -> Result<(), IpcError> {
        let process = self.processes.get(&process_pid)
            .ok_or(IpcError::NotFound)?;
        let audience = process.address_hash;

        let resource = alloc::format!("pid:{}", target_pid);
        let cap = self.identity.issue_root_token(
            entropy,
            &audience,
            CapabilityType::Endpoint,
            resource.as_bytes(),
            0,    // not_before: immediate
            now,  // expires_at: 0 = never (if now=0)
        ).map_err(|_| IpcError::PermissionDenied)?;

        let process = self.processes.get_mut(&process_pid)
            .ok_or(IpcError::NotFound)?;
        process.capabilities.push(cap);
        Ok(())
    }

    /// Check whether `capabilities` contain a valid EndpointCap for `target_pid`.
    pub(crate) fn check_endpoint_cap(
        &self,
        capabilities: &[UcanToken],
        audience_hash: &[u8; 16],
        target_pid: u32,
        now: u64,
    ) -> Result<(), IpcError> {
        let target_resource = alloc::format!("pid:{}", target_pid);

        for cap in capabilities {
            // Must be an Endpoint capability
            if cap.capability != CapabilityType::Endpoint {
                continue;
            }
            // Must be issued to this process
            if &cap.audience != audience_hash {
                continue;
            }
            // Resource must match target pid or be wildcard
            let resource_str = core::str::from_utf8(&cap.resource).unwrap_or("");
            if resource_str != target_resource && resource_str != "*" {
                continue;
            }
            // Cryptographic verification
            if verify_token(
                cap, now,
                &self.proof_store,
                &self.identity_store,
                &self.revocations,
                5, // max delegation depth
            ).is_ok() {
                return Ok(());
            }
        }

        Err(IpcError::PermissionDenied)
    }
}
```

**Step 4: Run tests to verify they pass**

```bash
cargo test -p harmony-microkernel
```

Expected: 23 tests pass (18 prior + 5 new).

**Step 5: Lint**

```bash
cargo clippy --workspace
```

**Step 6: Commit**

```bash
git add crates/harmony-microkernel/src/kernel.rs
git commit -m "feat(microkernel): Kernel struct with process management and capability enforcement"
```

---

## Task 5: Kernel IPC Dispatch

**Files:**
- Modify: `crates/harmony-microkernel/src/kernel.rs`

Add IPC dispatch methods to the Kernel. These resolve the caller's namespace,
check capabilities, and dispatch to the target's FileServer.

**Step 1: Write failing tests**

Add these tests to the `mod tests` block in `kernel.rs`:

```rust
    fn setup_kernel_with_echo() -> (Kernel, u32, u32) {
        let mut entropy = make_test_entropy();
        let kernel_id = PrivateIdentity::generate(&mut entropy);
        let mut kernel = Kernel::new(kernel_id);

        // pid 0 = echo server
        let server_pid = kernel.spawn_process(
            "echo-server",
            Box::new(EchoServer::new()),
            &[],
        );

        // pid 1 = client (also an echo server, but we don't use its server)
        let client_pid = kernel.spawn_process(
            "client",
            Box::new(EchoServer::new()),
            &[("/echo", server_pid, 0)],  // mount echo server
        );

        // Grant client access to server
        kernel.grant_endpoint_cap(&mut entropy, client_pid, server_pid, 0).unwrap();

        (kernel, client_pid, server_pid)
    }

    #[test]
    fn ipc_walk_through_namespace() {
        let (mut kernel, client, _server) = setup_kernel_with_echo();
        let qpath = kernel.walk(client, "/echo/hello", 0, 1, 0).unwrap();
        assert_eq!(qpath, 1); // hello's qpath
    }

    #[test]
    fn ipc_read_through_namespace() {
        let (mut kernel, client, _server) = setup_kernel_with_echo();
        kernel.walk(client, "/echo/hello", 0, 1, 0).unwrap();
        kernel.open(client, 1, OpenMode::Read, 0).unwrap();
        let data = kernel.read(client, 1, 0, 256, 0).unwrap();
        assert_eq!(data, b"Hello from echo server!");
    }

    #[test]
    fn ipc_write_and_read_echo() {
        let (mut kernel, client, _server) = setup_kernel_with_echo();
        kernel.walk(client, "/echo/echo", 0, 1, 0).unwrap();
        kernel.open(client, 1, OpenMode::ReadWrite, 0).unwrap();
        kernel.write(client, 1, 0, b"round trip", 0).unwrap();
        let data = kernel.read(client, 1, 0, 256, 0).unwrap();
        assert_eq!(data, b"round trip");
    }

    #[test]
    fn ipc_denied_without_capability() {
        let mut entropy = make_test_entropy();
        let kernel_id = PrivateIdentity::generate(&mut entropy);
        let mut kernel = Kernel::new(kernel_id);

        let server_pid = kernel.spawn_process(
            "echo-server",
            Box::new(EchoServer::new()),
            &[],
        );
        // Client has mount but NO capability
        let client_pid = kernel.spawn_process(
            "client",
            Box::new(EchoServer::new()),
            &[("/echo", server_pid, 0)],
        );

        assert_eq!(
            kernel.walk(client_pid, "/echo/hello", 0, 1, 0),
            Err(IpcError::PermissionDenied)
        );
    }

    #[test]
    fn ipc_unmounted_path() {
        let (mut kernel, client, _server) = setup_kernel_with_echo();
        assert_eq!(
            kernel.walk(client, "/nonexistent/file", 0, 1, 0),
            Err(IpcError::NotFound)
        );
    }

    #[test]
    fn ipc_clunk_then_read_fails() {
        let (mut kernel, client, _server) = setup_kernel_with_echo();
        kernel.walk(client, "/echo/hello", 0, 1, 0).unwrap();
        kernel.open(client, 1, OpenMode::Read, 0).unwrap();
        kernel.clunk(client, 1, 0).unwrap();
        assert_eq!(kernel.read(client, 1, 0, 256, 0), Err(IpcError::InvalidFid));
    }
```

**Step 2: Run tests to verify they fail**

```bash
cargo test -p harmony-microkernel
```

Expected: FAIL — method `walk` not found in `Kernel`.

**Step 3: Implement IPC dispatch methods**

Add these methods to the `impl Kernel` block:

```rust
    /// Walk a path on behalf of `from_pid`. Resolves the namespace,
    /// checks capabilities, and dispatches to the target FileServer.
    pub fn walk(
        &mut self,
        from_pid: u32,
        path: &str,
        _root_fid: Fid,
        new_fid: Fid,
        now: u64,
    ) -> Result<QPath, IpcError> {
        let process = self.processes.get(&from_pid).ok_or(IpcError::NotFound)?;
        let (mount, remainder) = process.namespace.resolve(path)
            .ok_or(IpcError::NotFound)?;
        let target_pid = mount.target_pid;
        let server_root_fid = mount.root_fid;

        // Capability check
        let caps = process.capabilities.clone();
        let addr = process.address_hash;
        self.check_endpoint_cap(&caps, &addr, target_pid, now)?;

        // Walk on the target server: from root fid to remainder
        let target = self.processes.get_mut(&target_pid)
            .ok_or(IpcError::NotFound)?;

        let qpath = if remainder.is_empty() {
            // Walking to the mount root itself — just return its root qpath
            target.server.stat(server_root_fid)
                .map(|st| st.qpath)?
        } else {
            target.server.walk(server_root_fid, new_fid, remainder)?
        };

        // Record fid ownership
        self.fid_owners.insert((from_pid, new_fid), target_pid);
        Ok(qpath)
    }

    /// Open a previously walked fid.
    pub fn open(
        &mut self,
        from_pid: u32,
        fid: Fid,
        mode: OpenMode,
        _now: u64,
    ) -> Result<(), IpcError> {
        let &target_pid = self.fid_owners.get(&(from_pid, fid))
            .ok_or(IpcError::InvalidFid)?;
        let target = self.processes.get_mut(&target_pid)
            .ok_or(IpcError::NotFound)?;
        target.server.open(fid, mode)
    }

    /// Read from a previously opened fid.
    pub fn read(
        &mut self,
        from_pid: u32,
        fid: Fid,
        offset: u64,
        count: u32,
        _now: u64,
    ) -> Result<Vec<u8>, IpcError> {
        let &target_pid = self.fid_owners.get(&(from_pid, fid))
            .ok_or(IpcError::InvalidFid)?;
        let target = self.processes.get_mut(&target_pid)
            .ok_or(IpcError::NotFound)?;
        target.server.read(fid, offset, count)
    }

    /// Write to a previously opened fid.
    pub fn write(
        &mut self,
        from_pid: u32,
        fid: Fid,
        offset: u64,
        data: &[u8],
        _now: u64,
    ) -> Result<u32, IpcError> {
        let &target_pid = self.fid_owners.get(&(from_pid, fid))
            .ok_or(IpcError::InvalidFid)?;
        let target = self.processes.get_mut(&target_pid)
            .ok_or(IpcError::NotFound)?;
        target.server.write(fid, offset, data)
    }

    /// Release a fid.
    pub fn clunk(
        &mut self,
        from_pid: u32,
        fid: Fid,
        _now: u64,
    ) -> Result<(), IpcError> {
        let target_pid = self.fid_owners.remove(&(from_pid, fid))
            .ok_or(IpcError::InvalidFid)?;
        let target = self.processes.get_mut(&target_pid)
            .ok_or(IpcError::NotFound)?;
        target.server.clunk(fid)
    }
```

**Step 4: Run tests to verify they pass**

```bash
cargo test -p harmony-microkernel
```

Expected: 29 tests pass (23 prior + 6 new).

**Step 5: Lint**

```bash
cargo clippy --workspace
```

**Step 6: Commit**

```bash
git add crates/harmony-microkernel/src/kernel.rs
git commit -m "feat(microkernel): IPC dispatch — walk/open/read/write/clunk through namespace"
```

---

## Task 6: End-to-End Integration Test

**Files:**
- Modify: `crates/harmony-microkernel/src/kernel.rs` (add integration test)

A comprehensive test that exercises the full IPC path: kernel creation,
process spawning, capability delegation, namespace resolution, and
multi-step IPC operations.

**Step 1: Write the integration test**

Add to the `mod tests` block in `kernel.rs`:

```rust
    #[test]
    fn integration_two_processes_full_ipc() {
        let mut entropy = make_test_entropy();
        let kernel_id = PrivateIdentity::generate(&mut entropy);
        let mut kernel = Kernel::new(kernel_id);

        // Spawn echo server as pid 0
        let server_pid = kernel.spawn_process(
            "echo-server",
            Box::new(EchoServer::new()),
            &[],
        );

        // Spawn client as pid 1, with echo server mounted at /svc/echo
        let client_pid = kernel.spawn_process(
            "harmony-node",
            Box::new(EchoServer::new()),  // client also serves, but we test as client
            &[("/svc/echo", server_pid, 0)],
        );

        // Grant client capability to access echo server
        kernel.grant_endpoint_cap(&mut entropy, client_pid, server_pid, 0).unwrap();

        // -- Full IPC sequence: walk → open → read → clunk --

        // 1. Walk to /svc/echo/hello
        let qpath = kernel.walk(client_pid, "/svc/echo/hello", 0, 1, 0).unwrap();
        assert_eq!(qpath, 1); // hello qpath

        // 2. Open for reading
        kernel.open(client_pid, 1, OpenMode::Read, 0).unwrap();

        // 3. Read the greeting
        let data = kernel.read(client_pid, 1, 0, 256, 0).unwrap();
        assert_eq!(data, b"Hello from echo server!");

        // 4. Clunk the fid
        kernel.clunk(client_pid, 1, 0).unwrap();

        // -- Echo round-trip: walk → open → write → read → clunk --

        // 5. Walk to /svc/echo/echo
        kernel.walk(client_pid, "/svc/echo/echo", 0, 2, 0).unwrap();

        // 6. Open for read/write
        kernel.open(client_pid, 2, OpenMode::ReadWrite, 0).unwrap();

        // 7. Write data
        let written = kernel.write(client_pid, 2, 0, b"Harmony Ring 2!", 0).unwrap();
        assert_eq!(written, 15);

        // 8. Read it back
        let data = kernel.read(client_pid, 2, 0, 256, 0).unwrap();
        assert_eq!(data, b"Harmony Ring 2!");

        // 9. Clunk
        kernel.clunk(client_pid, 2, 0).unwrap();

        // Verify clunked fids are invalid
        assert_eq!(kernel.read(client_pid, 1, 0, 256, 0), Err(IpcError::InvalidFid));
        assert_eq!(kernel.read(client_pid, 2, 0, 256, 0), Err(IpcError::InvalidFid));
    }
```

**Step 2: Run test**

```bash
cargo test -p harmony-microkernel integration_two_processes_full_ipc
```

Expected: PASS (all infrastructure from Tasks 1–5 is in place).

**Step 3: Lint**

```bash
cargo clippy --workspace
```

**Step 4: Commit**

```bash
git add crates/harmony-microkernel/src/kernel.rs
git commit -m "test(microkernel): end-to-end integration test — two processes over IPC"
```

---

## Task 7: KernelSerialServer

**Files:**
- Modify: `crates/harmony-microkernel/src/serial_server.rs`

A FileServer that captures writes to a buffer. In tests, we assert
on the buffer contents. In the boot crate (Task 8), this wraps the
real serial port.

**Step 1: Write failing tests**

Add to `crates/harmony-microkernel/src/serial_server.rs`:

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! KernelSerialServer — serial output as a FileServer.
//!
//! Exposes a single writable file `log`. Write bytes to it and they
//! accumulate in an internal buffer (or go to a real serial port in
//! the boot crate).

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::{Fid, FileServer, FileStat, FileType, IpcError, OpenMode, QPath};

// Implementation goes here (Step 3)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walk_to_log() {
        let mut srv = SerialServer::new();
        let qpath = srv.walk(0, 1, "log").unwrap();
        assert_eq!(qpath, 1);
    }

    #[test]
    fn write_captures_data() {
        let mut srv = SerialServer::new();
        srv.walk(0, 1, "log").unwrap();
        srv.open(1, OpenMode::Write).unwrap();
        srv.write(1, 0, b"hello serial").unwrap();
        assert_eq!(srv.buffer(), b"hello serial");
    }

    #[test]
    fn multiple_writes_append() {
        let mut srv = SerialServer::new();
        srv.walk(0, 1, "log").unwrap();
        srv.open(1, OpenMode::Write).unwrap();
        srv.write(1, 0, b"hello ").unwrap();
        srv.write(1, 0, b"world").unwrap();
        assert_eq!(srv.buffer(), b"hello world");
    }

    #[test]
    fn read_returns_buffer() {
        let mut srv = SerialServer::new();
        srv.walk(0, 1, "log").unwrap();
        srv.open(1, OpenMode::ReadWrite).unwrap();
        srv.write(1, 0, b"data").unwrap();
        let data = srv.read(1, 0, 256).unwrap();
        assert_eq!(data, b"data");
    }

    #[test]
    fn walk_invalid_name() {
        let mut srv = SerialServer::new();
        assert_eq!(srv.walk(0, 1, "nonexistent"), Err(IpcError::NotFound));
    }
}
```

**Step 2: Run tests to verify they fail**

```bash
cargo test -p harmony-microkernel
```

Expected: FAIL — `SerialServer` not found.

**Step 3: Implement SerialServer**

Add the implementation above the `#[cfg(test)]` section:

```rust
const QPATH_ROOT: QPath = 0;
const QPATH_LOG: QPath = 1;

struct FidState {
    qpath: QPath,
    is_open: bool,
}

/// A FileServer that captures writes to a buffer.
///
/// In tests, call `buffer()` to inspect what was written.
/// In the boot crate, the buffer can be drained to a real serial port.
pub struct SerialServer {
    fids: BTreeMap<Fid, FidState>,
    buf: Vec<u8>,
}

impl SerialServer {
    pub fn new() -> Self {
        let mut fids = BTreeMap::new();
        fids.insert(0, FidState { qpath: QPATH_ROOT, is_open: false });
        SerialServer { fids, buf: Vec::new() }
    }

    /// Access the accumulated write buffer.
    pub fn buffer(&self) -> &[u8] {
        &self.buf
    }
}

impl FileServer for SerialServer {
    fn walk(&mut self, fid: Fid, new_fid: Fid, name: &str) -> Result<QPath, IpcError> {
        let state = self.fids.get(&fid).ok_or(IpcError::InvalidFid)?;
        if state.qpath != QPATH_ROOT {
            return Err(IpcError::NotDirectory);
        }
        if name != "log" {
            return Err(IpcError::NotFound);
        }
        self.fids.insert(new_fid, FidState { qpath: QPATH_LOG, is_open: false });
        Ok(QPATH_LOG)
    }

    fn open(&mut self, fid: Fid, _mode: OpenMode) -> Result<(), IpcError> {
        let state = self.fids.get_mut(&fid).ok_or(IpcError::InvalidFid)?;
        state.is_open = true;
        Ok(())
    }

    fn read(&mut self, fid: Fid, _offset: u64, count: u32) -> Result<Vec<u8>, IpcError> {
        let state = self.fids.get(&fid).ok_or(IpcError::InvalidFid)?;
        if !state.is_open {
            return Err(IpcError::NotOpen);
        }
        let end = core::cmp::min(self.buf.len(), count as usize);
        Ok(self.buf[..end].to_vec())
    }

    fn write(&mut self, fid: Fid, _offset: u64, data: &[u8]) -> Result<u32, IpcError> {
        let state = self.fids.get(&fid).ok_or(IpcError::InvalidFid)?;
        if !state.is_open {
            return Err(IpcError::NotOpen);
        }
        self.buf.extend_from_slice(data);
        Ok(data.len() as u32)
    }

    fn clunk(&mut self, fid: Fid) -> Result<(), IpcError> {
        self.fids.remove(&fid).ok_or(IpcError::InvalidFid)?;
        Ok(())
    }

    fn stat(&mut self, fid: Fid) -> Result<FileStat, IpcError> {
        let state = self.fids.get(&fid).ok_or(IpcError::InvalidFid)?;
        let (name, file_type, size) = match state.qpath {
            QPATH_ROOT => ("/", FileType::Directory, 0),
            QPATH_LOG => ("log", FileType::Regular, self.buf.len() as u64),
            _ => return Err(IpcError::NotFound),
        };
        Ok(FileStat {
            qpath: state.qpath,
            name: Arc::from(name),
            size,
            file_type,
        })
    }
}
```

**Step 4: Run tests to verify they pass**

```bash
cargo test -p harmony-microkernel
```

Expected: 35 tests pass (30 prior + 5 new).

**Step 5: Lint**

```bash
cargo clippy --workspace
```

**Step 6: Commit**

```bash
git add crates/harmony-microkernel/src/serial_server.rs
git commit -m "feat(microkernel): SerialServer — serial output as a FileServer"
```

---

## Task 8: Boot Integration (ring2 feature)

**Files:**
- Modify: `crates/harmony-boot/Cargo.toml`
- Modify: `crates/harmony-boot/src/main.rs`

Add a `ring2` feature to harmony-boot. When enabled, the boot sequence
creates a Kernel, spawns an echo server, runs an IPC demo, and logs
results to serial — proving Ring 2 works on real hardware (QEMU).

**Important:** harmony-boot is EXCLUDED from the workspace (it targets
`x86_64-unknown-none`). Build/test it separately:

```bash
cd crates/harmony-boot
cargo check --target x86_64-unknown-none --features ring2
```

**Step 1: Update harmony-boot/Cargo.toml**

Add the `ring2` feature and the microkernel dependency. The microkernel
must be imported WITHOUT std features (bare metal):

Add to the `[dependencies]` section:

```toml
harmony-microkernel = { path = "../harmony-microkernel", default-features = false, optional = true }
```

Add to `[features]`:

```toml
ring2 = ["dep:harmony-microkernel"]
```

**Note:** harmony-microkernel currently requires `std`. Before this step
can compile for bare metal, harmony-microkernel's `lib.rs` needs the
`#![cfg_attr(not(feature = "std"), no_std)]` gate and a `std` feature
flag (like harmony-unikernel has). Add this to harmony-microkernel:

Update `crates/harmony-microkernel/Cargo.toml` features:

```toml
[features]
default = ["std"]
std = [
    "harmony-unikernel/std",
    "harmony-crypto/std",
    "harmony-identity/std",
    "harmony-identity/test-utils",
    "harmony-platform/std",
]
```

Update `crates/harmony-microkernel/src/lib.rs` top:

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
```

And update the workspace `Cargo.toml` to set `default-features = false`
for harmony-microkernel in workspace deps (it's already there but check):

```toml
harmony-microkernel = { path = "crates/harmony-microkernel", default-features = false }
```

**Step 2: Add ring2 boot path to main.rs**

Add a `mod microkernel_boot;` or inline the ring2 boot sequence. The
simplest approach is a conditional block in `kernel_main`:

After the existing `#[cfg(feature = "qemu-test")]` block (line 328–329)
and before the event loop, add:

```rust
    #[cfg(feature = "ring2")]
    {
        use harmony_microkernel::kernel::Kernel;
        use harmony_microkernel::echo::EchoServer;

        serial.log("KERN", "Ring 2 microkernel mode");

        let mut kernel_entropy_2 = KernelEntropy::new(rdrand_fill);
        let kernel_id = PrivateIdentity::generate(&mut kernel_entropy_2);
        let mut kernel = Kernel::new(kernel_id);

        // Spawn echo server as pid 0
        let server_pid = kernel.spawn_process(
            "echo-server",
            alloc::boxed::Box::new(EchoServer::new()),
            &[],
        );
        let _ = writeln!(serial, "[PROC] pid={} name=echo-server", server_pid);

        // Spawn client as pid 1
        let client_pid = kernel.spawn_process(
            "harmony-node",
            alloc::boxed::Box::new(EchoServer::new()),
            &[("/echo", server_pid, 0)],
        );
        let _ = writeln!(serial, "[PROC] pid={} name=harmony-node", client_pid);

        // Grant capability
        kernel.grant_endpoint_cap(&mut kernel_entropy_2, client_pid, server_pid, 0)
            .expect("capability grant failed");
        let _ = writeln!(serial, "[CAP]  pid={} granted Endpoint to pid={}", client_pid, server_pid);

        // IPC demo: walk → open → read
        match kernel.walk(client_pid, "/echo/hello", 0, 1, 0) {
            Ok(_) => { let _ = writeln!(serial, "[IPC]  walk /echo/hello ok"); }
            Err(e) => { let _ = writeln!(serial, "[IPC]  walk failed: {:?}", e); }
        }
        match kernel.open(client_pid, 1, harmony_microkernel::OpenMode::Read, 0) {
            Ok(_) => { let _ = writeln!(serial, "[IPC]  open fid=1 Read ok"); }
            Err(e) => { let _ = writeln!(serial, "[IPC]  open failed: {:?}", e); }
        }
        match kernel.read(client_pid, 1, 0, 256, 0) {
            Ok(data) => {
                let msg = core::str::from_utf8(&data).unwrap_or("(non-utf8)");
                let _ = writeln!(serial, "[IPC]  read: \"{}\"", msg);
            }
            Err(e) => { let _ = writeln!(serial, "[IPC]  read failed: {:?}", e); }
        }
        let _ = kernel.clunk(client_pid, 1, 0);
        serial.log("KERN", "Ring 2 IPC demo complete");
    }
```

**Step 3: Verify it compiles**

```bash
cd crates/harmony-boot
cargo check --target x86_64-unknown-none --features ring2
```

Expected: compiles without errors. Note: `harmony-identity`'s test-utils
feature depends on `hashbrown` which must support no_std. If it doesn't
compile, the `test-utils` feature may need to be gated behind `std` in
the microkernel. In that case, the Kernel would use custom trait
implementations instead of `Memory*` stores for no_std builds. This is
acceptable follow-up work — the important thing is that `cargo test`
passes on the host.

**Step 4: Verify workspace tests still pass**

```bash
cd ../..  # back to harmony-os root
cargo test --workspace
cargo clippy --workspace
```

**Step 5: Commit**

```bash
git add crates/harmony-microkernel/ crates/harmony-boot/
git commit -m "feat(boot): ring2 feature — microkernel IPC demo on QEMU"
```

---

## Summary

| Task | What it builds | Tests added |
|------|---------------|-------------|
| 1 | Core types + FileServer trait | 1 (object safety) |
| 2 | EchoServer | 10 |
| 3 | Namespace | 7 |
| 4 | Kernel + capability enforcement | 5 |
| 5 | IPC dispatch | 6 |
| 6 | Integration test | 1 (comprehensive) |
| 7 | SerialServer | 5 |
| 8 | Boot integration | compile check |
| **Total** | | **~35 tests** |

After all tasks, the milestone A demo is: two processes communicating
over 9P-inspired IPC with UCAN capability enforcement, all sans-I/O
and testable on the host with `cargo test -p harmony-microkernel`.

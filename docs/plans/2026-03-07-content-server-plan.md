# ContentServer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a 9P FileServer in harmony-os that exposes content-addressed chunk/blob storage at `/store` with ingest, blob read, and chunk read sub-namespaces.

**Architecture:** Single `ContentServer` struct implementing `FileServer` from harmony-microkernel. In-memory `BTreeMap` storage for chunks and blob metadata. Uses harmony-athenaeum for chunking, addressing, dedup, and reassembly. Mounted via the kernel's existing spawn/mount/capability pipeline.

**Tech Stack:** Rust, harmony-athenaeum (from harmony core), harmony-microkernel (in harmony-os)

**Design doc:** `docs/plans/2026-03-07-content-server-design.md`

---

### Task 1: Add harmony-athenaeum dependency

**Files:**
- Modify: `crates/harmony-microkernel/Cargo.toml`

**Context:** The microkernel crate needs harmony-athenaeum for `Athenaeum`, `ChunkAddr`, and `sha256_hash`. The workspace already declares it as a git dependency. The microkernel uses feature-gated optional deps — athenaeum should follow the same pattern under the `kernel` feature.

**Step 1: Add the dependency**

In `crates/harmony-microkernel/Cargo.toml`, add harmony-athenaeum to `[dependencies]` and the `kernel` feature:

```toml
[dependencies]
harmony-unikernel = { workspace = true }
harmony-crypto = { workspace = true, optional = true }
harmony-identity = { workspace = true, optional = true }
harmony-platform = { workspace = true, optional = true }
harmony-athenaeum = { workspace = true, optional = true }
```

And in the `[features]` section, add it to the `kernel` feature list:

```toml
kernel = [
    "std",
    "dep:harmony-crypto", "harmony-crypto/std",
    "dep:harmony-identity", "harmony-identity/std", "harmony-identity/test-utils",
    "dep:harmony-platform", "harmony-platform/std",
    "dep:harmony-athenaeum",
]
```

**Step 2: Verify it compiles**

Run: `cargo check -p harmony-microkernel`
Expected: compiles with no errors

**Step 3: Commit**

```bash
git add crates/harmony-microkernel/Cargo.toml
git commit -m "feat(microkernel): add harmony-athenaeum dependency"
```

---

### Task 2: Create ContentServer struct and constructor

**Files:**
- Create: `crates/harmony-microkernel/src/content_server.rs`
- Modify: `crates/harmony-microkernel/src/lib.rs` (add `pub mod content_server;`)

**Context:** This task creates the data model from the design doc — `ContentServer`, `FidState`, `NodeKind`, QPath constants, and `ContentServer::new()`. No FileServer methods yet — just the types and constructor.

**Step 1: Write the failing test**

At the bottom of `content_server.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_content_server_has_root_fid() {
        let server = ContentServer::new();
        // Fid 0 should exist and point to root
        let stat = server.fids.get(&0).unwrap();
        assert_eq!(stat.qpath, ROOT);
        assert!(!stat.is_open);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-microkernel content_server::tests::new_content_server_has_root_fid`
Expected: FAIL — module doesn't exist yet

**Step 3: Write the implementation**

Create `crates/harmony-microkernel/src/content_server.rs`:

```rust
//! ContentServer — content-addressed 9P file server for Ring 2.
//!
//! Exposes chunk and blob storage at `/store` with sub-namespaces
//! for blobs, chunks, and ingest. See design doc:
//! `docs/plans/2026-03-07-content-server-design.md`

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;

use harmony_athenaeum::{Athenaeum, ChunkAddr, sha256_hash};

use crate::{Fid, QPath, OpenMode, FileType, FileStat, IpcError, FileServer};

// --- QPath constants ---
const ROOT: QPath = 0;
const BLOBS_DIR: QPath = 1;
const CHUNKS_DIR: QPath = 2;
const INGEST: QPath = 3;
/// Dynamic qpaths for blobs start here.
const BLOB_QPATH_BASE: QPath = 0x1000;
/// Dynamic qpaths for chunks start here.
const CHUNK_QPATH_BASE: QPath = 0x100_0000;

/// Which node a fid currently points to.
#[derive(Debug, Clone)]
enum NodeKind {
    Root,
    BlobsDir,
    ChunksDir,
    Ingest,
    Blob([u8; 32]),
    Chunk(ChunkAddr),
}

/// Per-fid session state.
#[derive(Debug, Clone)]
struct FidState {
    qpath: QPath,
    node: NodeKind,
    is_open: bool,
    mode: Option<OpenMode>,
}

/// Ingest buffer state — tracks whether finalization has occurred.
#[derive(Debug)]
enum IngestState {
    /// Accumulating write data.
    Writing(Vec<u8>),
    /// Finalized — response ready for one-shot read.
    Finalized(Vec<u8>),
    /// Response already read.
    Done,
}

/// Content-addressed chunk and blob store, exposed as a 9P file server.
pub struct ContentServer {
    /// Raw chunk data keyed by 32-bit address.
    chunks: BTreeMap<ChunkAddr, Vec<u8>>,
    /// Blob metadata: CID → Athenaeum (chunk mapping + blob size).
    blobs: BTreeMap<[u8; 32], Athenaeum>,
    /// Per-fid session state.
    fids: BTreeMap<Fid, FidState>,
    /// In-flight ingest buffers keyed by fid.
    ingest_buffers: BTreeMap<Fid, IngestState>,
    /// Next fid for server-side allocation (used by clone_fid).
    next_fid: Fid,
}

impl ContentServer {
    /// Create a new empty ContentServer with root fid 0.
    pub fn new() -> Self {
        let mut fids = BTreeMap::new();
        fids.insert(
            0,
            FidState {
                qpath: ROOT,
                node: NodeKind::Root,
                is_open: false,
                mode: None,
            },
        );
        Self {
            chunks: BTreeMap::new(),
            blobs: BTreeMap::new(),
            fids,
            ingest_buffers: BTreeMap::new(),
            next_fid: 100,
        }
    }

    /// Compute a stable qpath for a blob CID.
    fn blob_qpath(cid: &[u8; 32]) -> QPath {
        let bits = u32::from_le_bytes([cid[0], cid[1], cid[2], cid[3]]);
        BLOB_QPATH_BASE + bits as u64
    }

    /// Compute a stable qpath for a chunk address.
    fn chunk_qpath(addr: ChunkAddr) -> QPath {
        CHUNK_QPATH_BASE + addr.hash_bits() as u64
    }

    /// Number of stored blobs.
    pub fn blob_count(&self) -> usize {
        self.blobs.len()
    }

    /// Number of stored chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}
```

Add to `crates/harmony-microkernel/src/lib.rs`, next to the other `pub mod` lines:

```rust
#[cfg(feature = "kernel")]
pub mod content_server;
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-microkernel content_server::tests::new_content_server_has_root_fid`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-microkernel/src/content_server.rs crates/harmony-microkernel/src/lib.rs
git commit -m "feat(microkernel): add ContentServer struct and types"
```

---

### Task 3: Implement walk()

**Files:**
- Modify: `crates/harmony-microkernel/src/content_server.rs`

**Context:** Walk routes through the namespace tree. Root has three children (blobs, chunks, ingest). BlobsDir children are hex CIDs (looked up in `self.blobs`). ChunksDir children are hex chunk addresses (looked up in `self.chunks`). All leaf files have no children. Follow the EchoServer pattern exactly: validate source fid, reject duplicate new_fid, route by name, insert new fid in closed state.

**Step 1: Write the failing tests**

```rust
#[test]
fn walk_root_to_blobs() {
    let mut server = ContentServer::new();
    let qpath = server.walk(0, 1, "blobs").unwrap();
    assert_eq!(qpath, BLOBS_DIR);
}

#[test]
fn walk_root_to_chunks() {
    let mut server = ContentServer::new();
    let qpath = server.walk(0, 1, "chunks").unwrap();
    assert_eq!(qpath, CHUNKS_DIR);
}

#[test]
fn walk_root_to_ingest() {
    let mut server = ContentServer::new();
    let qpath = server.walk(0, 1, "ingest").unwrap();
    assert_eq!(qpath, INGEST);
}

#[test]
fn walk_root_not_found() {
    let mut server = ContentServer::new();
    assert_eq!(server.walk(0, 1, "nonexistent"), Err(IpcError::NotFound));
}

#[test]
fn walk_invalid_source_fid() {
    let mut server = ContentServer::new();
    assert_eq!(server.walk(99, 1, "blobs"), Err(IpcError::InvalidFid));
}

#[test]
fn walk_duplicate_new_fid() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "blobs").unwrap();
    assert_eq!(server.walk(0, 1, "chunks"), Err(IpcError::InvalidFid));
}

#[test]
fn walk_from_non_directory() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "ingest").unwrap();
    assert_eq!(server.walk(1, 2, "anything"), Err(IpcError::NotDirectory));
}

#[test]
fn walk_blobs_dir_missing_cid() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "blobs").unwrap();
    // Walk to a CID that doesn't exist
    let fake_cid = "aa".repeat(32); // 64 hex chars
    assert_eq!(server.walk(1, 2, &fake_cid), Err(IpcError::NotFound));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-microkernel content_server::tests::walk_`
Expected: FAIL — `walk` method doesn't exist

**Step 3: Write the implementation**

Add the `FileServer` impl block (we'll add methods incrementally):

```rust
impl FileServer for ContentServer {
    fn walk(&mut self, fid: Fid, new_fid: Fid, name: &str) -> Result<QPath, IpcError> {
        let state = self.fids.get(&fid).ok_or(IpcError::InvalidFid)?;
        let parent_node = state.node.clone();

        if self.fids.contains_key(&new_fid) {
            return Err(IpcError::InvalidFid);
        }

        let (qpath, node) = match &parent_node {
            NodeKind::Root => match name {
                "blobs" => (BLOBS_DIR, NodeKind::BlobsDir),
                "chunks" => (CHUNKS_DIR, NodeKind::ChunksDir),
                "ingest" => (INGEST, NodeKind::Ingest),
                _ => return Err(IpcError::NotFound),
            },
            NodeKind::BlobsDir => {
                // Parse 64-char hex CID
                let cid = parse_hex_cid(name).ok_or(IpcError::NotFound)?;
                if !self.blobs.contains_key(&cid) {
                    return Err(IpcError::NotFound);
                }
                (Self::blob_qpath(&cid), NodeKind::Blob(cid))
            }
            NodeKind::ChunksDir => {
                // Parse 8-char hex ChunkAddr
                let addr = parse_hex_addr(name).ok_or(IpcError::NotFound)?;
                if !self.chunks.contains_key(&addr) {
                    return Err(IpcError::NotFound);
                }
                (Self::chunk_qpath(addr), NodeKind::Chunk(addr))
            }
            // Leaf nodes are not directories
            _ => return Err(IpcError::NotDirectory),
        };

        self.fids.insert(
            new_fid,
            FidState {
                qpath,
                node,
                is_open: false,
                mode: None,
            },
        );
        Ok(qpath)
    }

    // Stub remaining methods so it compiles:
    fn open(&mut self, _fid: Fid, _mode: OpenMode) -> Result<(), IpcError> {
        Err(IpcError::NotSupported)
    }
    fn read(&mut self, _fid: Fid, _offset: u64, _count: u32) -> Result<Vec<u8>, IpcError> {
        Err(IpcError::NotSupported)
    }
    fn write(&mut self, _fid: Fid, _offset: u64, _data: &[u8]) -> Result<u32, IpcError> {
        Err(IpcError::NotSupported)
    }
    fn clunk(&mut self, _fid: Fid) -> Result<(), IpcError> {
        Err(IpcError::NotSupported)
    }
    fn stat(&mut self, _fid: Fid) -> Result<FileStat, IpcError> {
        Err(IpcError::NotSupported)
    }
    fn clone_fid(&mut self, _fid: Fid, _new_fid: Fid) -> Result<QPath, IpcError> {
        Err(IpcError::NotSupported)
    }
}

/// Parse a 64-character hex string into a 32-byte CID.
fn parse_hex_cid(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut cid = [0u8; 32];
    for i in 0..32 {
        cid[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(cid)
}

/// Parse an 8-character hex string into a ChunkAddr.
fn parse_hex_addr(s: &str) -> Option<ChunkAddr> {
    if s.len() != 8 {
        return None;
    }
    let raw = u32::from_str_radix(s, 16).ok()?;
    Some(ChunkAddr::from_raw_u32(raw))
}

/// Format a 32-byte CID as a 64-character lowercase hex string.
fn format_cid_hex(cid: &[u8; 32]) -> alloc::string::String {
    let mut s = alloc::string::String::with_capacity(64);
    for byte in cid {
        use core::fmt::Write;
        write!(s, "{:02x}", byte).unwrap();
    }
    s
}

/// Format a ChunkAddr as an 8-character lowercase hex string.
fn format_addr_hex(addr: ChunkAddr) -> alloc::string::String {
    use core::fmt::Write;
    let mut s = alloc::string::String::with_capacity(8);
    write!(s, "{:08x}", addr.to_raw_u32()).unwrap();
    s
}
```

**Note:** `ChunkAddr::from_raw_u32` and `ChunkAddr::to_raw_u32` may not exist in the public API. Check `addr.rs` — if they don't exist, use `ChunkAddr(raw)` for construction (the field is `pub(crate)`) and add a public accessor method to harmony-athenaeum. If blocked, use the `hash_bits()` method and reconstruct differently. The implementer should check the exact API and adapt.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-microkernel content_server::tests::walk_`
Expected: all 8 walk tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-microkernel/src/content_server.rs
git commit -m "feat(microkernel): implement ContentServer walk()"
```

---

### Task 4: Implement open, clunk, clone_fid, stat

**Files:**
- Modify: `crates/harmony-microkernel/src/content_server.rs`

**Context:** Standard 9P boilerplate methods. Follow EchoServer patterns exactly. Mode validation at open time: directories reject Write/ReadWrite, blob/chunk files reject Write/ReadWrite (read-only), ingest requires ReadWrite.

**Step 1: Write the failing tests**

```rust
#[test]
fn open_and_clunk_ingest() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "ingest").unwrap();
    server.open(1, OpenMode::ReadWrite).unwrap();
    server.clunk(1).unwrap();
}

#[test]
fn open_ingest_read_only_rejected() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "ingest").unwrap();
    assert_eq!(server.open(1, OpenMode::Read), Err(IpcError::PermissionDenied));
}

#[test]
fn open_directory_write_rejected() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "blobs").unwrap();
    assert_eq!(server.open(1, OpenMode::Write), Err(IpcError::IsDirectory));
}

#[test]
fn open_directory_read_ok() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "blobs").unwrap();
    server.open(1, OpenMode::Read).unwrap();
}

#[test]
fn double_open_rejected() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "ingest").unwrap();
    server.open(1, OpenMode::ReadWrite).unwrap();
    assert_eq!(server.open(1, OpenMode::ReadWrite), Err(IpcError::PermissionDenied));
}

#[test]
fn clunk_root_rejected() {
    let mut server = ContentServer::new();
    assert_eq!(server.clunk(0), Err(IpcError::PermissionDenied));
}

#[test]
fn clunk_invalid_fid() {
    let mut server = ContentServer::new();
    assert_eq!(server.clunk(99), Err(IpcError::InvalidFid));
}

#[test]
fn clone_fid_root() {
    let mut server = ContentServer::new();
    let qpath = server.clone_fid(0, 5).unwrap();
    assert_eq!(qpath, ROOT);
}

#[test]
fn stat_root() {
    let mut server = ContentServer::new();
    let stat = server.stat(0).unwrap();
    assert_eq!(stat.file_type, FileType::Directory);
    assert_eq!(&*stat.name, "store");
}

#[test]
fn stat_ingest() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "ingest").unwrap();
    let stat = server.stat(1).unwrap();
    assert_eq!(stat.file_type, FileType::Regular);
    assert_eq!(&*stat.name, "ingest");
}

#[test]
fn stat_blobs_dir() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "blobs").unwrap();
    let stat = server.stat(1).unwrap();
    assert_eq!(stat.file_type, FileType::Directory);
    assert_eq!(&*stat.name, "blobs");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-microkernel content_server::tests::open_ content_server::tests::clunk_ content_server::tests::clone_ content_server::tests::stat_`
Expected: FAIL — methods return `NotSupported`

**Step 3: Implement the methods**

Replace the stub methods in the `FileServer` impl:

```rust
fn open(&mut self, fid: Fid, mode: OpenMode) -> Result<(), IpcError> {
    let state = self.fids.get_mut(&fid).ok_or(IpcError::InvalidFid)?;
    if state.is_open {
        return Err(IpcError::PermissionDenied);
    }
    match &state.node {
        NodeKind::Root | NodeKind::BlobsDir | NodeKind::ChunksDir => {
            if matches!(mode, OpenMode::Write | OpenMode::ReadWrite) {
                return Err(IpcError::IsDirectory);
            }
        }
        NodeKind::Blob(_) | NodeKind::Chunk(_) => {
            if matches!(mode, OpenMode::Write | OpenMode::ReadWrite) {
                return Err(IpcError::ReadOnly);
            }
        }
        NodeKind::Ingest => {
            if matches!(mode, OpenMode::Read | OpenMode::Write) {
                return Err(IpcError::PermissionDenied);
            }
            // Initialize ingest buffer
            self.ingest_buffers.insert(fid, IngestState::Writing(Vec::new()));
        }
    }
    state.is_open = true;
    state.mode = Some(mode);
    Ok(())
}

fn clunk(&mut self, fid: Fid) -> Result<(), IpcError> {
    if fid == 0 {
        return Err(IpcError::PermissionDenied);
    }
    self.fids.remove(&fid).ok_or(IpcError::InvalidFid)?;
    self.ingest_buffers.remove(&fid);
    Ok(())
}

fn clone_fid(&mut self, fid: Fid, new_fid: Fid) -> Result<QPath, IpcError> {
    if self.fids.contains_key(&new_fid) {
        return Err(IpcError::InvalidFid);
    }
    let state = self.fids.get(&fid).ok_or(IpcError::InvalidFid)?;
    let qpath = state.qpath;
    let node = state.node.clone();
    self.fids.insert(
        new_fid,
        FidState {
            qpath,
            node,
            is_open: false,
            mode: None,
        },
    );
    Ok(qpath)
}

fn stat(&mut self, fid: Fid) -> Result<FileStat, IpcError> {
    let state = self.fids.get(&fid).ok_or(IpcError::InvalidFid)?;
    match &state.node {
        NodeKind::Root => Ok(FileStat {
            qpath: ROOT,
            name: Arc::from("store"),
            size: 0,
            file_type: FileType::Directory,
        }),
        NodeKind::BlobsDir => Ok(FileStat {
            qpath: BLOBS_DIR,
            name: Arc::from("blobs"),
            size: 0,
            file_type: FileType::Directory,
        }),
        NodeKind::ChunksDir => Ok(FileStat {
            qpath: CHUNKS_DIR,
            name: Arc::from("chunks"),
            size: 0,
            file_type: FileType::Directory,
        }),
        NodeKind::Ingest => Ok(FileStat {
            qpath: INGEST,
            name: Arc::from("ingest"),
            size: 0,
            file_type: FileType::Regular,
        }),
        NodeKind::Blob(cid) => {
            let ath = self.blobs.get(cid).ok_or(IpcError::NotFound)?;
            Ok(FileStat {
                qpath: Self::blob_qpath(cid),
                name: Arc::from(format_cid_hex(cid).as_str()),
                size: ath.blob_size as u64,
                file_type: FileType::Regular,
            })
        }
        NodeKind::Chunk(addr) => {
            let data = self.chunks.get(addr).ok_or(IpcError::NotFound)?;
            Ok(FileStat {
                qpath: Self::chunk_qpath(*addr),
                name: Arc::from(format_addr_hex(*addr).as_str()),
                size: data.len() as u64,
                file_type: FileType::Regular,
            })
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-microkernel content_server::tests`
Expected: all tests PASS (walk + open/clunk/clone/stat)

**Step 5: Commit**

```bash
git add crates/harmony-microkernel/src/content_server.rs
git commit -m "feat(microkernel): implement ContentServer open/clunk/clone_fid/stat"
```

---

### Task 5: Implement write() — ingest accumulation

**Files:**
- Modify: `crates/harmony-microkernel/src/content_server.rs`

**Context:** Only `/store/ingest` accepts writes. All other nodes reject with `ReadOnly` or `IsDirectory`. Writes accumulate data in `ingest_buffers`. Once finalized (by a read), further writes are rejected.

**Step 1: Write the failing tests**

```rust
#[test]
fn write_to_ingest_accumulates() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "ingest").unwrap();
    server.open(1, OpenMode::ReadWrite).unwrap();
    let written = server.write(1, 0, b"hello").unwrap();
    assert_eq!(written, 5);
    let written = server.write(1, 0, b" world").unwrap();
    assert_eq!(written, 6);
}

#[test]
fn write_to_directory_rejected() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "blobs").unwrap();
    server.open(1, OpenMode::Read).unwrap();
    assert_eq!(server.write(1, 0, b"data"), Err(IpcError::PermissionDenied));
}

#[test]
fn write_without_open_rejected() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "ingest").unwrap();
    assert_eq!(server.write(1, 0, b"data"), Err(IpcError::NotOpen));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-microkernel content_server::tests::write_`
Expected: FAIL

**Step 3: Implement write()**

```rust
fn write(&mut self, fid: Fid, _offset: u64, data: &[u8]) -> Result<u32, IpcError> {
    let state = self.fids.get(&fid).ok_or(IpcError::InvalidFid)?;
    if !state.is_open {
        return Err(IpcError::NotOpen);
    }
    if matches!(state.mode, Some(OpenMode::Read)) {
        return Err(IpcError::PermissionDenied);
    }
    match &state.node {
        NodeKind::Root | NodeKind::BlobsDir | NodeKind::ChunksDir => {
            Err(IpcError::IsDirectory)
        }
        NodeKind::Blob(_) | NodeKind::Chunk(_) => {
            Err(IpcError::ReadOnly)
        }
        NodeKind::Ingest => {
            let buf = self.ingest_buffers.get_mut(&fid).ok_or(IpcError::InvalidArgument)?;
            match buf {
                IngestState::Writing(ref mut v) => {
                    v.extend_from_slice(data);
                    let len = u32::try_from(data.len())
                        .map_err(|_| IpcError::ResourceExhausted)?;
                    Ok(len)
                }
                _ => Err(IpcError::InvalidArgument), // Already finalized
            }
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-microkernel content_server::tests::write_`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-microkernel/src/content_server.rs
git commit -m "feat(microkernel): implement ContentServer write() for ingest"
```

---

### Task 6: Implement read() — ingest finalization and blob/chunk reads

**Files:**
- Modify: `crates/harmony-microkernel/src/content_server.rs`

**Context:** This is the most complex method. Three behaviors:
1. **Ingest read** — finalizes: chunks the blob via `Athenaeum::from_blob`, stores chunks/metadata, returns 40-byte response (CID + chunk_count + blob_size). One-shot.
2. **Blob read** — reassembles the blob from chunks via `Athenaeum::reassemble`, slices by offset/count.
3. **Chunk read** — returns raw chunk data sliced by offset/count.

**Step 1: Write the failing tests**

```rust
#[test]
fn ingest_and_read_back_metadata() {
    let mut server = ContentServer::new();
    // Ingest a blob
    server.walk(0, 1, "ingest").unwrap();
    server.open(1, OpenMode::ReadWrite).unwrap();
    let blob_data = vec![0xABu8; 4096]; // One full chunk
    server.write(1, 0, &blob_data).unwrap();

    // Read triggers finalization
    let response = server.read(1, 0, 256).unwrap();
    assert_eq!(response.len(), 40); // 32 CID + 4 chunk_count + 4 blob_size

    // Parse response
    let cid: [u8; 32] = response[..32].try_into().unwrap();
    let chunk_count = u32::from_le_bytes(response[32..36].try_into().unwrap());
    let blob_size = u32::from_le_bytes(response[36..40].try_into().unwrap());

    assert_eq!(cid, sha256_hash(&blob_data));
    assert_eq!(chunk_count, 1);
    assert_eq!(blob_size, 4096);

    // Server should now have 1 blob and 1 chunk
    assert_eq!(server.blob_count(), 1);
    assert_eq!(server.chunk_count(), 1);
}

#[test]
fn ingest_second_read_returns_empty() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "ingest").unwrap();
    server.open(1, OpenMode::ReadWrite).unwrap();
    server.write(1, 0, &[0xCDu8; 100]).unwrap();
    let first = server.read(1, 0, 256).unwrap();
    assert_eq!(first.len(), 40);
    let second = server.read(1, 0, 256).unwrap();
    assert!(second.is_empty());
}

#[test]
fn ingest_read_before_write_rejected() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "ingest").unwrap();
    server.open(1, OpenMode::ReadWrite).unwrap();
    // No write — read should fail
    assert_eq!(server.read(1, 0, 256), Err(IpcError::InvalidArgument));
}

#[test]
fn read_blob_by_cid() {
    let mut server = ContentServer::new();
    // Ingest
    let blob_data = vec![0x42u8; 8000]; // ~2 chunks
    server.walk(0, 1, "ingest").unwrap();
    server.open(1, OpenMode::ReadWrite).unwrap();
    server.write(1, 0, &blob_data).unwrap();
    let response = server.read(1, 0, 256).unwrap();
    let cid_hex = format_cid_hex(&response[..32].try_into().unwrap());
    server.clunk(1).unwrap();

    // Walk to the blob and read it back
    server.walk(0, 2, "blobs").unwrap();
    server.walk(2, 3, &cid_hex).unwrap();
    server.open(3, OpenMode::Read).unwrap();
    let read_back = server.read(3, 0, 16384).unwrap();
    assert_eq!(read_back, blob_data);
}

#[test]
fn read_blob_with_offset() {
    let mut server = ContentServer::new();
    let blob_data = vec![0xEFu8; 4096];
    server.walk(0, 1, "ingest").unwrap();
    server.open(1, OpenMode::ReadWrite).unwrap();
    server.write(1, 0, &blob_data).unwrap();
    let response = server.read(1, 0, 256).unwrap();
    let cid_hex = format_cid_hex(&response[..32].try_into().unwrap());
    server.clunk(1).unwrap();

    server.walk(0, 2, "blobs").unwrap();
    server.walk(2, 3, &cid_hex).unwrap();
    server.open(3, OpenMode::Read).unwrap();
    let slice = server.read(3, 100, 50).unwrap();
    assert_eq!(slice.len(), 50);
    assert_eq!(slice, vec![0xEFu8; 50]);
}

#[test]
fn read_directory_returns_is_directory() {
    let mut server = ContentServer::new();
    server.walk(0, 1, "blobs").unwrap();
    server.open(1, OpenMode::Read).unwrap();
    assert_eq!(server.read(1, 0, 256), Err(IpcError::IsDirectory));
}

#[test]
fn ingest_dedup_same_blob() {
    let mut server = ContentServer::new();
    let blob_data = vec![0xAAu8; 4096];

    // Ingest first time
    server.walk(0, 1, "ingest").unwrap();
    server.open(1, OpenMode::ReadWrite).unwrap();
    server.write(1, 0, &blob_data).unwrap();
    server.read(1, 0, 256).unwrap();
    server.clunk(1).unwrap();
    assert_eq!(server.chunk_count(), 1);

    // Ingest same blob again
    server.walk(0, 2, "ingest").unwrap();
    server.open(2, OpenMode::ReadWrite).unwrap();
    server.write(2, 0, &blob_data).unwrap();
    server.read(2, 0, 256).unwrap();
    server.clunk(2).unwrap();
    assert_eq!(server.chunk_count(), 1); // No new chunks
    assert_eq!(server.blob_count(), 1);  // No new blob
}

#[test]
fn ingest_oversized_blob_rejected() {
    use harmony_athenaeum::MAX_BLOB_SIZE;
    let mut server = ContentServer::new();
    server.walk(0, 1, "ingest").unwrap();
    server.open(1, OpenMode::ReadWrite).unwrap();
    let big = vec![0u8; MAX_BLOB_SIZE + 1];
    server.write(1, 0, &big).unwrap();
    assert_eq!(server.read(1, 0, 256), Err(IpcError::ResourceExhausted));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-microkernel content_server::tests`
Expected: FAIL — read() returns `NotSupported`

**Step 3: Implement read()**

```rust
fn read(&mut self, fid: Fid, offset: u64, count: u32) -> Result<Vec<u8>, IpcError> {
    let state = self.fids.get(&fid).ok_or(IpcError::InvalidFid)?;
    if !state.is_open {
        return Err(IpcError::NotOpen);
    }
    if matches!(state.mode, Some(OpenMode::Write)) {
        return Err(IpcError::PermissionDenied);
    }

    match &state.node {
        NodeKind::Root | NodeKind::BlobsDir | NodeKind::ChunksDir => {
            Err(IpcError::IsDirectory)
        }
        NodeKind::Ingest => {
            self.read_ingest(fid)
        }
        NodeKind::Blob(cid) => {
            let cid = *cid;
            self.read_blob(&cid, offset, count)
        }
        NodeKind::Chunk(addr) => {
            let addr = *addr;
            self.read_chunk(&addr, offset, count)
        }
    }
}
```

Add helper methods on `ContentServer`:

```rust
impl ContentServer {
    /// Finalize ingest: chunk the blob, store everything, return metadata.
    fn read_ingest(&mut self, fid: Fid) -> Result<Vec<u8>, IpcError> {
        let buf = self.ingest_buffers.get_mut(&fid).ok_or(IpcError::InvalidArgument)?;

        match core::mem::replace(buf, IngestState::Done) {
            IngestState::Writing(data) => {
                if data.is_empty() {
                    *buf = IngestState::Done;
                    return Err(IpcError::InvalidArgument);
                }
                if data.len() > harmony_athenaeum::MAX_BLOB_SIZE {
                    *buf = IngestState::Done;
                    return Err(IpcError::ResourceExhausted);
                }
                let cid = sha256_hash(&data);

                if !self.blobs.contains_key(&cid) {
                    let ath = Athenaeum::from_blob(cid, &data)
                        .map_err(|_| IpcError::ResourceExhausted)?;

                    // Store chunks
                    for (i, &addr) in ath.chunks.iter().enumerate() {
                        self.chunks.entry(addr).or_insert_with(|| {
                            let chunk_start = i * 4096;
                            let chunk_end = core::cmp::min(chunk_start + 4096, data.len());
                            let raw = &data[chunk_start..chunk_end];
                            // Pad to chunk size
                            let padded_size = addr.size_bytes();
                            let mut padded = alloc::vec![0u8; padded_size];
                            padded[..raw.len()].copy_from_slice(raw);
                            padded
                        });
                    }

                    let chunk_count = ath.chunks.len() as u32;
                    let blob_size = ath.blob_size as u32;
                    self.blobs.insert(cid, ath);

                    // Build 40-byte response
                    let mut response = Vec::with_capacity(40);
                    response.extend_from_slice(&cid);
                    response.extend_from_slice(&chunk_count.to_le_bytes());
                    response.extend_from_slice(&blob_size.to_le_bytes());

                    *buf = IngestState::Finalized(response.clone());

                    // Actually, return the response now and mark as Done on next read
                    *buf = IngestState::Done;
                    Ok(response)
                } else {
                    // Blob already exists — return existing metadata
                    let ath = &self.blobs[&cid];
                    let mut response = Vec::with_capacity(40);
                    response.extend_from_slice(&cid);
                    response.extend_from_slice(&(ath.chunks.len() as u32).to_le_bytes());
                    response.extend_from_slice(&(ath.blob_size as u32).to_le_bytes());
                    *buf = IngestState::Done;
                    Ok(response)
                }
            }
            IngestState::Finalized(response) => {
                // Already consumed
                Ok(Vec::new())
            }
            IngestState::Done => {
                Ok(Vec::new())
            }
        }
    }

    /// Read blob data by reassembling from chunks.
    fn read_blob(&self, cid: &[u8; 32], offset: u64, count: u32) -> Result<Vec<u8>, IpcError> {
        let ath = self.blobs.get(cid).ok_or(IpcError::NotFound)?;
        let chunks = &self.chunks;
        let data = ath.reassemble(|addr| chunks.get(&addr).cloned())
            .map_err(|_| IpcError::NotFound)?;
        slice_data(&data, offset, count)
    }

    /// Read raw chunk data.
    fn read_chunk(&self, addr: &ChunkAddr, offset: u64, count: u32) -> Result<Vec<u8>, IpcError> {
        let data = self.chunks.get(addr).ok_or(IpcError::NotFound)?;
        slice_data(data, offset, count)
    }
}

/// Standard 9P offset/count slicing.
fn slice_data(data: &[u8], offset: u64, count: u32) -> Result<Vec<u8>, IpcError> {
    let offset = offset.min(usize::MAX as u64) as usize;
    if offset >= data.len() {
        return Ok(Vec::new());
    }
    let end = core::cmp::min(offset.saturating_add(count as usize), data.len());
    Ok(data[offset..end].to_vec())
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-microkernel content_server::tests`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-microkernel/src/content_server.rs
git commit -m "feat(microkernel): implement ContentServer read() with ingest finalization"
```

---

### Task 7: Integration test through the Kernel

**Files:**
- Modify: `crates/harmony-microkernel/src/content_server.rs` (add integration test)
  OR
- Modify: `crates/harmony-microkernel/src/kernel.rs` (add test at bottom)

**Context:** Test the full IPC flow: spawn ContentServer as a process, mount it, grant capability, ingest a blob through the kernel's walk/open/write/read/clunk pipeline, then walk to the blob by CID and read it back through the kernel.

**Step 1: Write the failing test**

Add to `kernel.rs` tests (where other integration tests live):

```rust
#[test]
fn content_server_ingest_and_read_via_kernel() {
    use crate::content_server::ContentServer;

    let kernel_id = PrivateIdentity::generate(test_rng());
    let mut kernel = Kernel::new(kernel_id);
    let mut entropy = test_rng();

    // Spawn content server
    let server_pid = kernel
        .spawn_process("content-store", Box::new(ContentServer::new()), &[])
        .unwrap();

    // Spawn client with /store mounted to content server
    let client_pid = kernel
        .spawn_process(
            "test-client",
            Box::new(EchoServer::new()),
            &[("/store", server_pid, 0)],
        )
        .unwrap();

    // Grant capability
    kernel
        .grant_endpoint_cap(&mut entropy, client_pid, server_pid, 0)
        .unwrap();

    // Walk to /store/ingest and open for ReadWrite
    kernel.walk(client_pid, "/store/ingest", 0, 1, 0).unwrap();
    kernel.open(client_pid, 1, OpenMode::ReadWrite).unwrap();

    // Write blob data
    let blob_data = vec![0x42u8; 4096];
    let written = kernel.write(client_pid, 1, 0, &blob_data).unwrap();
    assert_eq!(written, 4096);

    // Read to finalize — get back CID + metadata
    let response = kernel.read(client_pid, 1, 0, 256).unwrap();
    assert_eq!(response.len(), 40);

    let mut cid = [0u8; 32];
    cid.copy_from_slice(&response[..32]);
    let cid_hex = crate::content_server::tests::format_cid_hex_pub(&cid);

    kernel.clunk(client_pid, 1).unwrap();

    // Now walk to /store/blobs/<cid> and read the blob back
    let blob_path = alloc::format!("/store/blobs/{}", cid_hex);
    kernel.walk(client_pid, &blob_path, 0, 2, 0).unwrap();
    kernel.open(client_pid, 2, OpenMode::Read).unwrap();

    let read_back = kernel.read(client_pid, 2, 0, 16384).unwrap();
    assert_eq!(read_back, blob_data);

    kernel.clunk(client_pid, 2).unwrap();
}
```

**Note:** The `format_cid_hex` function is private in `content_server.rs`. The implementer should either make it `pub(crate)` or add a small public helper. Adapt as needed.

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-microkernel content_server_ingest_and_read_via_kernel`
Expected: FAIL (won't compile yet, or test logic fails)

**Step 3: Fix any compilation issues and make test pass**

The main issues to resolve:
- `format_cid_hex` visibility (make `pub(crate)`)
- Ensure `ContentServer` is exported from the `content_server` module
- Any `ChunkAddr` API mismatches (from_raw_u32/to_raw_u32)

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-microkernel content_server_ingest_and_read_via_kernel`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-microkernel/src/content_server.rs crates/harmony-microkernel/src/kernel.rs
git commit -m "test(microkernel): add ContentServer kernel integration test"
```

---

### Task 8: Final quality gates

**Files:**
- All modified files

**Step 1: Run full test suite**

Run: `cargo test --workspace`
Expected: all tests PASS (existing 87 + new ContentServer tests)

**Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: zero warnings

**Step 3: Run fmt**

Run: `cargo fmt --all -- --check`
Expected: no diffs (run `cargo fmt --all` to fix if needed)

**Step 4: Commit any fmt fixes**

```bash
git add -A
git commit -m "style(microkernel): apply cargo fmt"
```

**Step 5: Verify test count**

Run: `cargo test --workspace 2>&1 | grep "test result"`
Expected: all suites show 0 failed

---

## Summary

| Task | What | Tests |
|------|------|-------|
| 1 | Add athenaeum dependency | compile check |
| 2 | ContentServer struct + types + constructor | 1 test |
| 3 | walk() implementation | 8 tests |
| 4 | open/clunk/clone_fid/stat | 11 tests |
| 5 | write() for ingest | 3 tests |
| 6 | read() — ingest finalization + blob/chunk reads | 8 tests |
| 7 | Kernel integration test | 1 test |
| 8 | Quality gates | full suite |

**Total new tests:** ~32

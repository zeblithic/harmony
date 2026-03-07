# ContentServer — 9P Content-Addressed Storage for Ring 2

**Date:** 2026-03-07
**Status:** Approved
**Scope:** harmony-os (crates/harmony-microkernel)

## Problem

The Ring 2 microkernel has a 9P IPC layer with capability-gated file servers (echo, serial), but no content-addressed storage. Now that harmony-athenaeum lives in harmony core with Encyclopedia support, Ring 2 needs a file server that exposes chunk storage and blob retrieval through the standard 9P namespace.

Processes need to:
1. Ingest blobs and get back content-addressed identifiers (CIDs)
2. Retrieve blobs by CID (reassembled from chunks)
3. Access raw chunks by address for low-level operations (delta sync, migration)

## Design

### Architecture

A single `ContentServer` struct implements `FileServer`. It manages both blob metadata and chunk data in-memory, exposed through a hierarchical 9P namespace at `/store`.

No kernel changes required — ContentServer is mounted like any other file server via the existing spawn + mount + capability pipeline.

### Namespace Layout

```
/store/                    (directory — root)
├── blobs/                 (directory)
│   └── <cid_hex>/         (read-only file — 64-char hex CID)
├── chunks/                (directory)
│   └── <addr_hex>/        (read-only file — 8-char hex ChunkAddr)
└── ingest                 (read-write file — ctl-file pattern)
```

### Data Model

```rust
pub struct ContentServer {
    /// Raw chunk data keyed by 32-bit address.
    chunks: BTreeMap<ChunkAddr, Vec<u8>>,
    /// Blob metadata: CID → Athenaeum (chunk mapping + blob size).
    blobs: BTreeMap<[u8; 32], Athenaeum>,
    /// Per-fid session state.
    fids: BTreeMap<Fid, FidState>,
    /// In-flight ingest buffers (fid → accumulated blob data).
    ingest_buffers: BTreeMap<Fid, Vec<u8>>,
    /// Next server-side fid for allocation.
    next_fid: Fid,
}

struct FidState {
    qpath: QPath,
    node: NodeKind,
    open: Option<OpenMode>,
}

enum NodeKind {
    Root,              // /store
    BlobsDir,          // /store/blobs
    ChunksDir,         // /store/chunks
    Ingest,            // /store/ingest
    Blob([u8; 32]),    // /store/blobs/<cid>
    Chunk(ChunkAddr),  // /store/chunks/<addr>
}
```

**QPath assignments:** Root=0, BlobsDir=1, ChunksDir=2, Ingest=3. Dynamic blob/chunk qpaths derived from content hash to remain stable across opens.

### Walk Rules

| From | Name | Target |
|------|------|--------|
| Root | `"blobs"` | BlobsDir |
| Root | `"chunks"` | ChunksDir |
| Root | `"ingest"` | Ingest |
| BlobsDir | `<64-char hex CID>` | Blob — `NotFound` if CID not in `self.blobs` |
| ChunksDir | `<8-char hex addr>` | Chunk — `NotFound` if addr not in `self.chunks` |
| Blob/Chunk/Ingest | any | `NotFound` (leaf files) |

### Ingest Flow (ctl-file pattern)

The ingest file uses the Plan 9 write-then-read pattern:

1. Client opens `/store/ingest` in `ReadWrite` mode
2. Client writes blob data (multiple writes accumulate in `ingest_buffers[fid]`)
3. Each write returns bytes written (standard 9P)
4. Client reads from the same fid — triggers finalization:
   - Validate size ≤ `MAX_BLOB_SIZE` (1MB)
   - Compute CID via `sha256_hash(&data)`
   - If CID exists → skip chunking (blob-level dedup)
   - Otherwise → `Athenaeum::from_blob(cid, &data)`, store chunks and metadata
   - Return 40-byte response: `cid (32) + chunk_count (4 LE) + blob_size (4 LE)`
5. Subsequent reads return empty (one-shot response)

**Edge cases:**
- Read before write → `InvalidArgument`
- Write after read → `InvalidArgument` (re-open for another ingest)
- Clunk always clears the ingest buffer for that fid

### Read/Write Semantics

| Path | Open modes | Read | Write | Stat |
|------|-----------|------|-------|------|
| `/store` | Read | `IsDirectory` | rejected | name="store", type=Dir |
| `/store/blobs` | Read | `IsDirectory` | rejected | name="blobs", type=Dir |
| `/store/chunks` | Read | `IsDirectory` | rejected | name="chunks", type=Dir |
| `/store/blobs/<cid>` | Read | reassemble from chunks, slice by offset/count | `ReadOnly` | name=hex CID, size=blob_size |
| `/store/chunks/<addr>` | Read | raw chunk data, slice by offset/count | `ReadOnly` | name=hex addr, size=chunk_size |
| `/store/ingest` | ReadWrite | finalize + return metadata | accumulate blob data | name="ingest", size=0 |

**Blob reassembly:** Uses `Athenaeum::reassemble()` with a closure over `self.chunks`. O(n) in chunk count — for a 1MB blob that's 256 BTreeMap lookups, fast enough for in-memory storage.

### Kernel Integration

```rust
let content_server = ContentServer::new();
let content_pid = kernel.spawn_process(Box::new(content_server), now);
kernel.grant_endpoint_cap(entropy, client_pid, content_pid, now);
// In client's namespace:
// mount("/store", content_pid, root_fid=0)
```

Uses `CapabilityType::Endpoint` for access gating. The `Content` capability type (value 5) exists in harmony-identity but is not enforced yet — per-CID access control is a future refinement.

### Storage

In-memory only (`BTreeMap`). No persistence, no trait abstraction. Matches the pattern of existing Ring 2 servers (echo, serial). A `ChunkStore` trait can be extracted when a second backend materializes.

## Testing

- **Unit tests on ContentServer:** walk/open/read/write/stat/clunk for each node type
- **Ingest round-trip:** write blob → read response → walk to blob by CID → read back → verify match
- **Chunk access:** after ingest, walk to individual chunks by address, verify data
- **Dedup:** ingest same blob twice, verify chunk count doesn't increase
- **Error cases:** read missing CID, write to blob file, oversized ingest, read before write on ingest
- **Integration through Kernel:** spawn ContentServer, mount, ingest via IPC, read back via IPC

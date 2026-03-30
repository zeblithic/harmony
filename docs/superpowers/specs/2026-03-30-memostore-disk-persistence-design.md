# MemoStore Disk Persistence

**Date:** 2026-03-30
**Status:** Draft
**Bead:** harmony-dm3t

## Problem

MemoStore is currently in-memory only (`HashMap<ContentId, Vec<Memo>>`). On node restart, all memos are lost and must be re-fetched from the network. For the build cache use case and content-addressed archival, memos are WORM attestations of deterministic computation that should persist indefinitely — they're knowledge, not ephemeral state.

## Constraints

- **Follow `disk_io.rs` pattern** — flat files, write-to-temp-then-rename, hex prefix sharding, synchronous I/O via `spawn_blocking`. Zero new dependencies.
- **postcard serialization** — already implemented in `harmony_memo::serialize/deserialize` with version byte prefix.
- **WORM semantics** — memos don't expire as a storage management lever. `expires_at` in the credential is for verification only, not eviction.
- **LFU eviction** — every node manages its own storage via least-frequently-used eviction. When full, evict LFU; offload to archivist if available, otherwise burn.
- **No_std compatible** — LFU tracking in `harmony-memo` crate uses `hashbrown::HashMap`.

## Architecture

### On-Disk Layout

Memos stored as individual postcard-serialized files alongside existing book storage:

```
{data_dir}/
├── book/                              # Existing content books
│   └── {prefix}/
│       └── {hex_cid}
└── memo/                              # New: memo persistence
    └── {prefix}/                      # 2-char hex prefix (input CID bytes [4..5])
        └── {input}_{output}_{signer}  # Postcard-serialized Memo
```

Filenames encode the dedup key `(input, output, signer_hash)` as hex. The `signer_hash` is `hex(memo.credential.issuer.hash)` (16 bytes = 32 hex chars). The 2-char prefix directory sharding prevents single-directory inode pressure (same pattern as books).

LFU access counts stored in `{data_dir}/memo_lfu.bin` — a postcard-serialized `HashMap<(ContentId, ContentId, [u8; 16]), u32>`. Written periodically (every 5 minutes) and on graceful shutdown, not on every access.

### memo_io.rs — Disk Operations

New file `harmony-node/src/memo_io.rs` mirroring `disk_io.rs`:

```rust
pub fn memo_path(data_dir: &Path, memo: &Memo) -> PathBuf
    // {data_dir}/memo/{input_hex[8..10]}/{input_hex}_{output_hex}_{signer_hex}

pub fn write_memo(data_dir: &Path, memo: &Memo) -> Result<(), io::Error>
    // Serialize with harmony_memo::serialize()
    // Write to temp file, sync_all(), rename to final path (atomic)

pub fn read_memo(data_dir: &Path, path: &Path) -> Result<Memo, io::Error>
    // Read bytes, harmony_memo::deserialize()

pub fn scan_memos(data_dir: &Path) -> Vec<Memo>
    // Walk {data_dir}/memo/, deserialize each file
    // Skip invalid files with warnings (same as scan_books)

pub fn delete_memo(data_dir: &Path, memo: &Memo) -> Result<(), io::Error>
    // Remove file by constructed path
```

All synchronous — called inside `tokio::task::spawn_blocking` from the event loop.

### MemoStore LFU Extension

The in-memory `MemoStore` in `harmony-memo/src/store.rs` gains LFU tracking via a parallel HashMap:

```rust
// New field in MemoStore
lfu_counts: HashMap<(ContentId, ContentId, [u8; 16]), u32>,
```

**Incremented on**: every `get_by_input` and `get_by_input_and_signer` call — each returned memo's counter bumps by 1.

**Eviction**: `evict_lfu(&mut self) -> Option<Memo>` finds the memo with the lowest access count, removes it from both `by_input` and `lfu_counts`, returns it. The caller decides whether to offload (publish to archivist Zenoh topic) or burn.

**LFU persistence**: `lfu_counts` serialized to `memo_lfu.bin` via postcard. Loaded at startup. Missing or corrupt file → all counters start at 0 (safe degradation).

### Event Loop Integration

Three integration points in `harmony-node/src/event_loop.rs`:

**On startup** (after config parse, before entering the loop):
- If `data_dir` is set, `scan_memos()` via `spawn_blocking`
- Insert each loaded memo into in-memory `MemoStore`
- Load `memo_lfu.bin` into the store's LFU counters
- Log: `"Loaded {n} memos from disk"`

**On memo insert** (Zenoh subscription or CLI):
- Insert into in-memory `MemoStore`
- If `data_dir` is set, dispatch `write_memo()` via `spawn_blocking`
- If total memo disk usage exceeds `disk_quota`:
  - `evict_lfu()` → delete file → optionally publish to `harmony/content/archive/*`
  - Log: `"Evicted memo {input}→{output} (LFU count: {n})"`

**Periodic tick** (every 5 minutes, piggybacks on existing timer):
- Serialize `lfu_counts` to `memo_lfu.bin` via `spawn_blocking`

**On graceful shutdown** (SIGTERM):
- Flush `lfu_counts` to `memo_lfu.bin` one final time

**Disk quota tracking**: Running `memo_disk_bytes: u64` counter, incremented on write (serialized size), decremented on delete. Initialized from `scan_memos` file sizes at startup. No filesystem stat calls during runtime.

## Testing Strategy

| Test | What | Location |
|---|---|---|
| `memo_io_write_read_roundtrip` | Write memo to disk, read back, compare | `harmony-node/src/memo_io.rs` |
| `memo_io_scan_discovers_memos` | Write 3 memos, scan directory, verify all found | `harmony-node/src/memo_io.rs` |
| `memo_io_delete_removes_file` | Write then delete, verify file gone | `harmony-node/src/memo_io.rs` |
| `memo_io_invalid_file_skipped` | Put garbage file in memo dir, scan doesn't panic | `harmony-node/src/memo_io.rs` |
| `lfu_counts_increment_on_query` | Insert, query, verify count = 1 | `harmony-memo/src/store.rs` |
| `evict_lfu_returns_lowest` | Insert 3, query unevenly, evict returns lowest | `harmony-memo/src/store.rs` |
| `evict_lfu_empty_returns_none` | Empty store eviction returns None | `harmony-memo/src/store.rs` |
| `lfu_counts_serialize_roundtrip` | Serialize LFU map, deserialize, compare | `harmony-memo/src/store.rs` |
| `quota_enforcement_triggers_eviction` | Small quota, insert until exceeded, verify eviction | integration test |

## Out of Scope

- **Archivist coordination** — offload-to-archivist is a Zenoh publish; the archivist subscription and replication logic is tracked in separate beads (harmony-os-ps1, harmony-swlq).
- **Memo Zenoh runtime integration** — wiring MemoStore into Zenoh subscriptions for receiving/publishing memos is a separate concern. This bead covers only disk persistence and LFU for the store itself.
- **Time-based expiry eviction** — memos are WORM. `expires_at` is for verification, not storage management.
- **Content book LFU** — books have their own eviction via `StorageTier`. Memos get their own parallel LFU tracking.

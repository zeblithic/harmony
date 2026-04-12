# harmony-db Phase 1 Design — Reusable CAS Database Abstraction

## Goal

Extract the "CAS + local index" pattern proven in harmony-client's mail module into a
reusable `harmony-db` crate. Any Harmony application can use it for structured key-value
storage with content-addressed snapshots, atomic commits, history diffing, and portable
index rebuild from CAS.

Phase 1 uses sorted in-memory pages with JSON persistence. The API is designed so Phase 2
can replace internals with Prolly Trees without changing the public contract.

## Location

`harmony/crates/harmony-db/` — alongside harmony-content and other shared infrastructure.

## Dependencies

- `harmony-content` — ContentId, BookStore trait, MemoryBookStore (for tests)
- `blake3` — CID computation for value blobs and commit snapshots
- `serde`, `serde_json` — index and commit serialization
- `hex` — CID hex encoding for filenames and JSON
- `thiserror` — error types (already in harmony ecosystem)

No embedded database engine. Local index is sorted JSON pages in memory.

## Data Model

### HarmonyDb

```rust
pub struct HarmonyDb {
    data_dir: PathBuf,
    tables: HashMap<String, Table>,
    head: Option<ContentId>,   // CID of the most recent commit
}
```

Top-level handle. Owns the data directory, all named tables, and the current head commit
reference. Opened via `open()` or `open_from_cas()`.

### Table

```rust
struct Table {
    entries: Vec<Entry>,       // sorted by key, always
}
```

A named collection of sorted key-value entries. Binary search for point lookups, slice for
range scans. Insertion maintains sort order via `Vec::insert` at the bisected position.
O(n) inserts, O(log n) lookups — adequate for <100K entries per table.

### Entry

```rust
pub struct Entry {
    pub key: Vec<u8>,          // arbitrary bytes, sorted lexicographically
    pub value_cid: ContentId,  // BLAKE3 hash of the value bytes
    pub timestamp: u64,        // insertion time (UNIX seconds)
    pub metadata: EntryMeta,   // consumer-defined flags
}
```

Keys are raw bytes — consumers encode their own format. Mail might use
`{timestamp_be}{cid_hex}` for time-ordered listing. Follows might use address bytes.
Values are stored as blobs, referenced by CID. The index holds only the reference, keeping
memory footprint small (~200 bytes per entry).

### EntryMeta

```rust
pub struct EntryMeta {
    pub flags: u64,            // bitfield for consumer use (e.g., bit 0 = read)
    pub snippet: String,       // short summary (max 256 bytes, for list views)
}
```

Consumer-extensible metadata stored in the index alongside each entry. `flags` is opaque
to harmony-db — consumers define bit semantics. `snippet` is a short string for list views
(subject lines, contact names); harmony-db truncates to 256 bytes on insert to bound index
size. Queryable without fetching the value blob.

## Public API

### Lifecycle

```rust
/// Open or create a database at `data_dir`.
/// Loads the local index cache if present; otherwise starts empty.
pub fn open(data_dir: &Path) -> Result<Self, DbError>;

/// Open, then immediately rebuild the index from a CAS commit.
/// Used for bootstrapping a new device from a synced root CID.
pub fn open_from_cas(data_dir: &Path, root: ContentId,
                     store: &impl BookStore) -> Result<Self, DbError>;
```

### Table Operations

```rust
/// Get or create a named table. Returns a mutable reference.
pub fn table(&mut self, name: &str) -> &mut Table;

/// List all table names.
pub fn table_names(&self) -> Vec<&str>;
```

### Table Methods

```rust
impl Table {
    /// Insert a key-value pair. Computes BLAKE3 CID for the value,
    /// stores the blob, upserts the entry (replaces if key exists).
    pub fn insert(&mut self, key: &[u8], value: &[u8], meta: EntryMeta,
                  blobs_dir: &Path) -> Result<ContentId, DbError>;

    /// Get the value bytes for a key. Reads the blob from disk.
    pub fn get(&self, key: &[u8], blobs_dir: &Path)
               -> Result<Option<Vec<u8>>, DbError>;

    /// Get just the entry (metadata + CID) without reading the blob.
    pub fn get_entry(&self, key: &[u8]) -> Option<&Entry>;

    /// Ordered iteration over a key range [start, end).
    /// Returns entries only — call get() to fetch value blobs.
    pub fn range(&self, start: &[u8], end: &[u8]) -> &[Entry];

    /// All entries in sorted order.
    pub fn entries(&self) -> &[Entry];

    /// Remove an entry by key. Returns the removed entry if found.
    /// Does NOT delete the value blob (other tables may reference it).
    pub fn remove(&mut self, key: &[u8]) -> Option<Entry>;

    /// Update metadata for an entry without touching the value.
    pub fn update_meta(&mut self, key: &[u8], meta: EntryMeta)
                       -> Result<(), DbError>;

    /// Number of entries.
    pub fn len(&self) -> usize;
}
```

Insert is an upsert — same key replaces the entry. Remove doesn't delete blobs because
the same CID may be referenced by multiple tables (e.g., self-sent mail appears in both
inbox and sent). Range returns a zero-copy slice into the sorted vec.

Note: Table methods show `blobs_dir: &Path` parameters in the signatures above. In
implementation, Table will hold an internal reference to the database's blobs directory
so callers don't need to pass it. The signatures here show the data flow; the ergonomic
API will be `table.insert(key, value, meta)` without the path parameter.

### Commit / History

```rust
/// Atomic snapshot: serialize all tables to CAS blobs, produce a root
/// manifest, persist locally. Optionally pushes to a BookStore.
pub fn commit(&mut self, store: Option<&mut impl BookStore>)
              -> Result<ContentId, DbError>;

/// Current head commit CID (None if no commits yet).
pub fn head(&self) -> Option<ContentId>;

/// Diff two commits. Returns entries added, removed, or changed per table.
pub fn diff(&self, old: ContentId, new: ContentId) -> Result<Diff, DbError>;

/// Rebuild the in-memory index from a CAS commit snapshot.
/// Fetches blobs locally first, falls back to BookStore if provided.
pub fn rebuild_from(&mut self, root: ContentId,
                    store: Option<&impl BookStore>) -> Result<(), DbError>;
```

### Diff Types

```rust
pub struct Diff {
    pub tables: HashMap<String, TableDiff>,
}

pub struct TableDiff {
    pub added: Vec<Entry>,
    pub removed: Vec<Entry>,
    pub changed: Vec<(Entry, Entry)>,  // (old, new) — same key, different value_cid
}
```

## On-Disk Layout

```
data_dir/
├── index.json              # local cache: all tables + head CID
├── index.json.tmp          # atomic write staging (transient)
├── blobs/
│   ├── {value_cid_hex}.bin     # raw value bytes
│   └── {value_cid_hex}.bin.tmp # atomic write staging (transient)
└── commits/
    ├── {root_cid_hex}.bin      # serialized root manifest
    ├── {page_cid_hex}.bin      # serialized table pages
    └── ...
```

### index.json

```json
{
  "version": 1,
  "head": "ab3f...64hex...cdef",
  "tables": {
    "inbox": {
      "entries": [
        {
          "key": "0189abcd...",
          "value_cid": "fe29...",
          "timestamp": 1744480000,
          "metadata": { "flags": 1, "snippet": "Meeting tomorrow" }
        }
      ]
    }
  }
}
```

This is the derived cache — reconstructable from CAS commits via `rebuild_from()`.
Missing or corrupt index.json causes a fresh start (no crash). Keys and CIDs are
hex-encoded in JSON.

### Atomic write pattern

All file writes use tmp-then-rename:

1. Serialize data
2. Write to `{filename}.tmp`
3. `fs::rename("{filename}.tmp", "{filename}")` — atomic on POSIX
4. Errors propagate to caller

Save happens on every mutation (insert, remove, update_meta). Crash-safe — either the
old file or the new file exists, never a partial write.

### Commit snapshots

Stored in `commits/` as content-addressed blobs:

**Root manifest:**
```json
{
  "version": 1,
  "parent": "previous_root_cid_or_null",
  "tables": {
    "inbox": "page_cid_hex",
    "sent": "page_cid_hex"
  }
}
```

**Page blobs:** JSON-serialized sorted entry lists for a single table. One page per table
per commit. (Phase 2 replaces these with Prolly Tree chunks.)

The parent chain enables `diff` to walk history, and `rebuild_from` to find predecessor
states.

## Commit Flow

1. For each table: serialize entries to JSON → BLAKE3 hash → write to `commits/{page_cid}.bin`
2. Build root manifest: `{ version, parent: head, tables: { name → page_cid } }`
3. BLAKE3 hash manifest → write to `commits/{root_cid}.bin`
4. Update `self.head = Some(root_cid)`
5. Save index.json
6. If BookStore provided: push all new blobs (values + pages + manifest) to store

## Diff Algorithm

Per-table two-pointer merge over sorted entries:

1. Load old and new root manifests from `commits/`
2. For each table present in both: if page CID is identical, skip (O(1) short-circuit)
3. Otherwise, load both page blobs and two-pointer merge:
   - Key only in old → removed
   - Key only in new → added
   - Key in both, different value_cid → changed
4. Tables only in new → all entries added
5. Tables only in old → all entries removed

The page CID short-circuit makes diff O(d) in practice — proportional to what changed.

## Rebuild Flow

1. Fetch root manifest: try `commits/{root_cid}.bin` locally, fall back to BookStore
2. For each table: fetch page blob (local → BookStore fallback), deserialize entries
3. For each entry: verify value blob exists in `blobs/`
   - Missing + BookStore available: fetch and save locally
   - Missing + no BookStore: keep entry but mark as orphaned (warn, don't fail)
4. Replace `self.tables` with rebuilt state
5. Set `self.head = root_cid`
6. Save index.json

Orphaned entries don't fail the rebuild. The entry appears in list views (you can see
metadata/snippet) but `get()` returns `None`. Resilient to partial sync — headers are
visible before bodies arrive.

## Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialize(String),

    #[error("corrupt index: {0}")]
    CorruptIndex(String),

    #[error("commit not found: {cid}")]
    CommitNotFound { cid: String },

    #[error("table not found: {name}")]
    TableNotFound { name: String },

    #[error("value blob missing: {cid}")]
    BlobMissing { cid: String },
}
```

Each variant maps to a distinct recovery strategy. `CorruptIndex` → start fresh or
rebuild. `CommitNotFound` / `BlobMissing` → sync gap. `TableNotFound` → programmer error.

## Testing Strategy

### Unit tests

| Test | Validates |
|------|-----------|
| `open_empty` | Fresh DB creates dirs, no tables, no head |
| `open_existing` | Insert → save → reopen → entries preserved |
| `insert_and_get` | KV round-trip, metadata preserved |
| `insert_upsert` | Same key replaces entry and value |
| `sorted_order` | Lexicographic sort maintained after mixed inserts |
| `range_query` | Correct slice returned, empty range is empty |
| `remove_entry` | Remove returns entry, get returns None after |
| `update_meta` | Metadata changes without affecting value_cid |
| `multi_table` | Operations on separate tables are independent |
| `commit_produces_cid` | Commit returns ContentId, head updated |
| `diff_no_changes` | Diff same commit → empty diff |
| `diff_additions` | New entries show as added |
| `diff_removals` | Removed entries show as removed |
| `diff_changes` | Updated values show old and new |
| `diff_table_skip` | Unchanged table has identical page CID, skipped |
| `rebuild_round_trip` | Commit → rebuild → identical state |
| `rebuild_from_bookstore` | Commit to BookStore, delete local commits, rebuild |
| `rebuild_orphaned_blobs` | Missing value blobs → entries present, get returns None |
| `corrupt_index_recovery` | Corrupt index.json → open starts empty |
| `atomic_write_crash_safety` | .tmp files don't corrupt existing state |
| `blob_dedup` | Same value twice → same CID, one blob file |

### Integration example (`examples/mail_workflow.rs`)

Mimics the mail.rs lifecycle without touching mail.rs:

- Create DB with inbox/sent/drafts/trash tables
- Receive 5 messages → insert into inbox
- Mark 2 as read → update_meta
- Move 1 to trash → remove from inbox, insert into trash
- Send a message → insert into sent
- Commit → get root CID
- Diff against previous → shows changes
- Rebuild from root CID → state matches
- Diff pre-move vs post-move → shows the moved message

## Scope Boundaries

**In scope (this phase):**
- HarmonyDb struct with open, table, commit, head, diff, rebuild_from
- Table with insert, get, get_entry, range, entries, remove, update_meta, len
- JSON-based persistence with atomic writes
- BLAKE3 content addressing for values and commits
- Optional BookStore push on commit / pull on rebuild
- Unit tests + mail workflow example

**Out of scope (future phases):**
- Secondary indexes (Phase 2 — Prolly Trees make these natural)
- Blob garbage collection (consumer-managed for now)
- Concurrent write access / locking (consumers wrap in Arc<Mutex<>> as needed)
- Network sync protocol (Zenoh integration is consumer-level)
- Migration tooling for existing mail.rs/follows.rs data

## Future: Phase 2 Transition

The sorted `Vec<Entry>` inside each Table becomes a Prolly Tree. The public API is
unchanged — `insert`, `get`, `range` work the same way. Commit pages become Prolly Tree
chunks with content-defined boundaries. Diff becomes native tree comparison instead of
two-pointer merge. The on-disk layout gains chunked page files instead of one-page-per-table.
The `commits/` directory structure and root manifest format remain compatible.

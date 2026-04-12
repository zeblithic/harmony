# harmony-db Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable CAS-backed key-value database crate with named tables, atomic commits, history diffing, and portable index rebuild.

**Architecture:** Sorted in-memory pages with JSON persistence. Content-addressed via harmony-content's ContentId. Local filesystem for blobs and commits, optional BookStore push for network sync. API designed so Phase 2 can swap internals to Prolly Trees without changing the public contract.

**Tech Stack:** Rust, harmony-content (ContentId, BookStore, MemoryBookStore), serde/serde_json, hex, thiserror

---

## File Structure

```
harmony/crates/harmony-db/
├── Cargo.toml          # Crate manifest, workspace deps
├── src/
│   ├── lib.rs          # Public API re-exports, module declarations
│   ├── error.rs        # DbError enum
│   ├── types.rs        # Entry, EntryMeta, Diff, TableDiff, serde helpers
│   ├── table.rs        # Table — sorted vec operations (no I/O)
│   ├── persist.rs      # Atomic file I/O, index load/save, blob read/write
│   ├── db.rs           # HarmonyDb — orchestrates tables + persistence
│   └── commit.rs       # Commit snapshots, diff, rebuild
└── examples/
    └── mail_workflow.rs # Integration example mimicking mail.rs patterns
```

**Responsibilities:**
- `error.rs` — Single error enum, all variants, Display/Error impls via thiserror
- `types.rs` — Pure data structs (Entry, EntryMeta, Diff, TableDiff) + serde helpers for hex-encoding ContentId and `Vec<u8>` keys in JSON
- `table.rs` — Sorted `Vec<Entry>` with binary search. No filesystem access. Pure logic.
- `persist.rs` — All filesystem I/O: atomic tmp+rename writes, index.json load/save, blob read/write. Separated so table.rs tests don't need tempdir.
- `db.rs` — Public HarmonyDb struct. Owns tables + data_dir. Methods delegate to table ops + persist for I/O. Calls `persist::save_index` after every mutation.
- `commit.rs` — Commit serialization (tables → page blobs → root manifest), diff (two-pointer merge), rebuild (CAS → local state). Uses persist module for blob I/O.

---

### Task 1: Scaffold Crate and Register in Workspace

**Files:**
- Create: `crates/harmony-db/Cargo.toml`
- Create: `crates/harmony-db/src/lib.rs`
- Modify: `Cargo.toml` (workspace root — add member + dep)

- [ ] **Step 1: Create crate directory**

```bash
mkdir -p crates/harmony-db/src crates/harmony-db/examples
```

- [ ] **Step 2: Create Cargo.toml**

Create `crates/harmony-db/Cargo.toml`:

```toml
[package]
name = "harmony-db"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Content-addressed key-value database with atomic commits and history diffing"

[dependencies]
harmony-content = { workspace = true, features = ["std"] }
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true, features = ["std"] }
hex = { workspace = true }
thiserror = { workspace = true, features = ["std"] }

[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 3: Create initial lib.rs**

Create `crates/harmony-db/src/lib.rs`:

```rust
//! Content-addressed key-value database with named tables, atomic commits,
//! history diffing, and portable index rebuild from CAS.

mod error;
mod types;

pub use error::DbError;
pub use types::{Entry, EntryMeta};
```

- [ ] **Step 4: Create stub error.rs**

Create `crates/harmony-db/src/error.rs`:

```rust
/// Errors returned by harmony-db operations.
#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
```

- [ ] **Step 5: Create stub types.rs**

Create `crates/harmony-db/src/types.rs`:

```rust
use harmony_content::ContentId;
use serde::{Deserialize, Serialize};

/// Consumer-extensible metadata stored alongside each entry in the index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EntryMeta {
    /// Opaque bitfield for consumer use (e.g., bit 0 = read).
    pub flags: u64,
    /// Short summary for list views (truncated to 256 bytes on insert).
    pub snippet: String,
}

/// A single key-value pair in a table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Entry {
    /// Arbitrary bytes, sorted lexicographically.
    #[serde(with = "hex_bytes")]
    pub key: Vec<u8>,
    /// BLAKE3 content address of the value bytes.
    #[serde(with = "hex_content_id")]
    pub value_cid: ContentId,
    /// Insertion time (UNIX seconds).
    pub timestamp: u64,
    /// Consumer-defined metadata.
    pub metadata: EntryMeta,
}

mod hex_bytes {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        hex::decode(&s).map_err(serde::de::Error::custom)
    }
}

mod hex_content_id {
    use harmony_content::ContentId;
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(cid: &ContentId, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(cid.to_bytes()))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ContentId, D::Error> {
        let s = String::deserialize(d)?;
        let bytes: [u8; 32] = hex::decode(&s)
            .map_err(serde::de::Error::custom)?
            .try_into()
            .map_err(|_| serde::de::Error::custom("expected 64 hex chars for ContentId"))?;
        Ok(ContentId::from_bytes(bytes))
    }
}
```

- [ ] **Step 6: Register in workspace**

Add `"crates/harmony-db",` to the workspace members list in `Cargo.toml` (after `harmony-content`). Add `harmony-db = { path = "crates/harmony-db" }` to `[workspace.dependencies]`.

- [ ] **Step 7: Verify it compiles**

```bash
cargo check -p harmony-db
```

Expected: compiles with no errors.

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-db/ Cargo.toml
git commit -m "feat(harmony-db): scaffold crate with error and type stubs (ZEB-97)"
```

---

### Task 2: Error Types

**Files:**
- Modify: `crates/harmony-db/src/error.rs`

- [ ] **Step 1: Write failing tests for error display**

Add to `crates/harmony-db/src/error.rs`:

```rust
/// Errors returned by harmony-db operations.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_error_display() {
        let err: DbError = std::io::Error::new(std::io::ErrorKind::NotFound, "gone").into();
        assert!(err.to_string().contains("I/O error"));
    }

    #[test]
    fn serialize_error_display() {
        let err = DbError::Serialize("bad json".into());
        assert_eq!(err.to_string(), "serialization error: bad json");
    }

    #[test]
    fn commit_not_found_display() {
        let err = DbError::CommitNotFound { cid: "abc123".into() };
        assert_eq!(err.to_string(), "commit not found: abc123");
    }

    #[test]
    fn table_not_found_display() {
        let err = DbError::TableNotFound { name: "inbox".into() };
        assert_eq!(err.to_string(), "table not found: inbox");
    }

    #[test]
    fn blob_missing_display() {
        let err = DbError::BlobMissing { cid: "deadbeef".into() };
        assert_eq!(err.to_string(), "value blob missing: deadbeef");
    }

    #[test]
    fn corrupt_index_display() {
        let err = DbError::CorruptIndex("truncated".into());
        assert_eq!(err.to_string(), "corrupt index: truncated");
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p harmony-db -- error
```

Expected: all 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-db/src/error.rs
git commit -m "feat(harmony-db): complete error types with tests (ZEB-97)"
```

---

### Task 3: Table — Sorted Vec Operations

**Files:**
- Create: `crates/harmony-db/src/table.rs`
- Modify: `crates/harmony-db/src/lib.rs` (add module)

Table is a pure data structure — sorted `Vec<Entry>` with binary search. No filesystem I/O. This makes it easy to test in isolation.

- [ ] **Step 1: Write failing test for insert + get_entry**

Create `crates/harmony-db/src/table.rs`:

```rust
use crate::types::{Entry, EntryMeta};
use harmony_content::{ContentFlags, ContentId};

/// A named collection of sorted key-value entries.
///
/// Maintains lexicographic order by key. Uses binary search for lookups
/// and slice views for range scans. No filesystem I/O — pure data structure.
#[derive(Debug, Clone)]
pub(crate) struct Table {
    pub(crate) entries: Vec<Entry>,
}

impl Table {
    pub fn new() -> Self {
        Table { entries: Vec::new() }
    }

    /// Insert or replace an entry. Maintains sorted order by key.
    /// Returns the previous entry if the key already existed.
    pub fn upsert(&mut self, entry: Entry) -> Option<Entry> {
        match self.entries.binary_search_by(|e| e.key.cmp(&entry.key)) {
            Ok(idx) => {
                let old = std::mem::replace(&mut self.entries[idx], entry);
                Some(old)
            }
            Err(idx) => {
                self.entries.insert(idx, entry);
                None
            }
        }
    }

    /// Look up an entry by key.
    pub fn get_entry(&self, key: &[u8]) -> Option<&Entry> {
        let idx = self.entries.binary_search_by(|e| e.key.as_slice().cmp(key)).ok()?;
        Some(&self.entries[idx])
    }

    /// Remove an entry by key. Returns the removed entry if found.
    pub fn remove(&mut self, key: &[u8]) -> Option<Entry> {
        let idx = self.entries.binary_search_by(|e| e.key.as_slice().cmp(key)).ok()?;
        Some(self.entries.remove(idx))
    }

    /// Update metadata for an existing entry. Returns false if key not found.
    pub fn update_meta(&mut self, key: &[u8], meta: EntryMeta) -> bool {
        if let Ok(idx) = self.entries.binary_search_by(|e| e.key.as_slice().cmp(key)) {
            self.entries[idx].metadata = meta;
            true
        } else {
            false
        }
    }

    /// Ordered slice over a key range [start, end).
    pub fn range(&self, start: &[u8], end: &[u8]) -> &[Entry] {
        let lo = match self.entries.binary_search_by(|e| e.key.as_slice().cmp(start)) {
            Ok(i) | Err(i) => i,
        };
        let hi = match self.entries.binary_search_by(|e| e.key.as_slice().cmp(end)) {
            Ok(i) | Err(i) => i,
        };
        &self.entries[lo..hi]
    }

    /// All entries in sorted order.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(key: &[u8], flags: u64, snippet: &str) -> Entry {
        let cid = ContentId::for_book(key, ContentFlags::default()).unwrap();
        Entry {
            key: key.to_vec(),
            value_cid: cid,
            timestamp: 1000,
            metadata: EntryMeta { flags, snippet: snippet.to_string() },
        }
    }

    #[test]
    fn insert_and_get_entry() {
        let mut t = Table::new();
        let e = make_entry(b"hello", 0, "greet");
        assert!(t.upsert(e.clone()).is_none());
        let found = t.get_entry(b"hello").unwrap();
        assert_eq!(found.metadata.snippet, "greet");
    }

    #[test]
    fn upsert_replaces_existing() {
        let mut t = Table::new();
        t.upsert(make_entry(b"key", 0, "old"));
        let old = t.upsert(make_entry(b"key", 1, "new"));
        assert!(old.is_some());
        assert_eq!(old.unwrap().metadata.snippet, "old");
        assert_eq!(t.get_entry(b"key").unwrap().metadata.snippet, "new");
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn sorted_order_maintained() {
        let mut t = Table::new();
        t.upsert(make_entry(b"cherry", 0, ""));
        t.upsert(make_entry(b"apple", 0, ""));
        t.upsert(make_entry(b"banana", 0, ""));
        let keys: Vec<&[u8]> = t.entries().iter().map(|e| e.key.as_slice()).collect();
        assert_eq!(keys, vec![b"apple".as_slice(), b"banana", b"cherry"]);
    }

    #[test]
    fn remove_entry() {
        let mut t = Table::new();
        t.upsert(make_entry(b"a", 0, ""));
        t.upsert(make_entry(b"b", 0, ""));
        let removed = t.remove(b"a");
        assert!(removed.is_some());
        assert!(t.get_entry(b"a").is_none());
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut t = Table::new();
        assert!(t.remove(b"nope").is_none());
    }

    #[test]
    fn update_meta() {
        let mut t = Table::new();
        t.upsert(make_entry(b"key", 0, "old"));
        let ok = t.update_meta(b"key", EntryMeta { flags: 1, snippet: "new".into() });
        assert!(ok);
        assert_eq!(t.get_entry(b"key").unwrap().metadata.flags, 1);
    }

    #[test]
    fn update_meta_nonexistent_returns_false() {
        let mut t = Table::new();
        assert!(!t.update_meta(b"nope", EntryMeta { flags: 0, snippet: "".into() }));
    }

    #[test]
    fn range_query() {
        let mut t = Table::new();
        t.upsert(make_entry(b"a", 0, ""));
        t.upsert(make_entry(b"b", 0, ""));
        t.upsert(make_entry(b"c", 0, ""));
        t.upsert(make_entry(b"d", 0, ""));
        let range = t.range(b"b", b"d");
        let keys: Vec<&[u8]> = range.iter().map(|e| e.key.as_slice()).collect();
        assert_eq!(keys, vec![b"b".as_slice(), b"c"]);
    }

    #[test]
    fn range_empty() {
        let mut t = Table::new();
        t.upsert(make_entry(b"a", 0, ""));
        assert!(t.range(b"x", b"z").is_empty());
    }

    #[test]
    fn get_entry_nonexistent() {
        let t = Table::new();
        assert!(t.get_entry(b"nope").is_none());
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

Update `crates/harmony-db/src/lib.rs`:

```rust
//! Content-addressed key-value database with named tables, atomic commits,
//! history diffing, and portable index rebuild from CAS.

mod error;
mod table;
mod types;

pub use error::DbError;
pub use types::{Entry, EntryMeta};
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p harmony-db -- table
```

Expected: all 10 table tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-db/src/table.rs crates/harmony-db/src/lib.rs
git commit -m "feat(harmony-db): sorted table with binary search and range queries (ZEB-97)"
```

---

### Task 4: Persistence — Atomic Writes, Index Load/Save, Blob I/O

**Files:**
- Create: `crates/harmony-db/src/persist.rs`
- Modify: `crates/harmony-db/src/lib.rs` (add module)

This module handles all filesystem I/O. Separated from table.rs so table tests stay fast and pure.

- [ ] **Step 1: Write the persistence module**

Create `crates/harmony-db/src/persist.rs`:

```rust
use crate::error::DbError;
use crate::table::Table;
use crate::types::{Entry, EntryMeta};
use harmony_content::{ContentFlags, ContentId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

const INDEX_VERSION: u32 = 1;
const MAX_SNIPPET_BYTES: usize = 256;

/// On-disk representation of the database index.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct IndexFile {
    pub version: u32,
    #[serde(
        serialize_with = "crate::types::ser_opt_cid",
        deserialize_with = "crate::types::de_opt_cid"
    )]
    pub head: Option<ContentId>,
    pub tables: HashMap<String, TableFile>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TableFile {
    pub entries: Vec<Entry>,
}

/// Load the index from `data_dir/index.json`.
/// Returns empty state if the file is missing or corrupt.
pub(crate) fn load_index(data_dir: &Path) -> (Option<ContentId>, HashMap<String, Table>) {
    let path = data_dir.join("index.json");
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return (None, HashMap::new()),
    };
    let idx: IndexFile = match serde_json::from_slice(&bytes) {
        Ok(i) if i.version == INDEX_VERSION => i,
        _ => return (None, HashMap::new()),
    };
    let tables = idx.tables.into_iter().map(|(name, tf)| {
        let mut table = Table::new();
        for entry in tf.entries {
            table.upsert(entry);
        }
        (name, table)
    }).collect();
    (idx.head, tables)
}

/// Atomically save the index to `data_dir/index.json`.
pub(crate) fn save_index(
    data_dir: &Path,
    head: Option<ContentId>,
    tables: &HashMap<String, Table>,
) -> Result<(), DbError> {
    let idx = IndexFile {
        version: INDEX_VERSION,
        head,
        tables: tables.iter().map(|(name, table)| {
            (name.clone(), TableFile { entries: table.entries().to_vec() })
        }).collect(),
    };
    let bytes = serde_json::to_vec_pretty(&idx)
        .map_err(|e| DbError::Serialize(e.to_string()))?;
    atomic_write(&data_dir.join("index.json"), &bytes)
}

/// Write a value blob to `data_dir/blobs/{cid_hex}.bin`.
/// Returns the ContentId. Idempotent — existing blobs are not overwritten.
pub(crate) fn write_blob(data_dir: &Path, value: &[u8]) -> Result<ContentId, DbError> {
    let cid = ContentId::for_book(value, ContentFlags::default())
        .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
    let cid_hex = hex::encode(cid.to_bytes());
    let blob_path = data_dir.join("blobs").join(format!("{cid_hex}.bin"));
    if blob_path.exists() {
        return Ok(cid);
    }
    let tmp = data_dir.join("blobs").join(format!("{cid_hex}.bin.tmp"));
    std::fs::write(&tmp, value)?;
    std::fs::rename(&tmp, &blob_path)?;
    Ok(cid)
}

/// Read a value blob by ContentId.
pub(crate) fn read_blob(data_dir: &Path, cid: &ContentId) -> Result<Option<Vec<u8>>, DbError> {
    let cid_hex = hex::encode(cid.to_bytes());
    let blob_path = data_dir.join("blobs").join(format!("{cid_hex}.bin"));
    match std::fs::read(&blob_path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(DbError::Io(e)),
    }
}

/// Ensure data_dir and its subdirectories exist.
pub(crate) fn ensure_dirs(data_dir: &Path) -> Result<(), DbError> {
    std::fs::create_dir_all(data_dir.join("blobs"))?;
    std::fs::create_dir_all(data_dir.join("commits"))?;
    Ok(())
}

/// Truncate a snippet to MAX_SNIPPET_BYTES on a UTF-8 boundary.
pub(crate) fn truncate_snippet(s: &str) -> String {
    if s.len() <= MAX_SNIPPET_BYTES {
        return s.to_string();
    }
    let mut end = MAX_SNIPPET_BYTES;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// Atomic write: write to tmp, then rename.
fn atomic_write(path: &Path, data: &[u8]) -> Result<(), DbError> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_and_read_blob() {
        let dir = tempfile::tempdir().unwrap();
        ensure_dirs(dir.path()).unwrap();
        let value = b"hello harmony-db";
        let cid = write_blob(dir.path(), value).unwrap();
        let read = read_blob(dir.path(), &cid).unwrap();
        assert_eq!(read.unwrap(), value);
    }

    #[test]
    fn blob_dedup_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        ensure_dirs(dir.path()).unwrap();
        let cid1 = write_blob(dir.path(), b"same").unwrap();
        let cid2 = write_blob(dir.path(), b"same").unwrap();
        assert_eq!(cid1, cid2);
    }

    #[test]
    fn read_missing_blob_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        ensure_dirs(dir.path()).unwrap();
        let cid = ContentId::for_book(b"nope", ContentFlags::default()).unwrap();
        assert!(read_blob(dir.path(), &cid).unwrap().is_none());
    }

    #[test]
    fn save_and_load_index_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        ensure_dirs(dir.path()).unwrap();
        let mut tables = HashMap::new();
        let mut t = Table::new();
        let cid = ContentId::for_book(b"val", ContentFlags::default()).unwrap();
        t.upsert(Entry {
            key: b"mykey".to_vec(),
            value_cid: cid,
            timestamp: 42,
            metadata: EntryMeta { flags: 1, snippet: "test".into() },
        });
        tables.insert("inbox".to_string(), t);
        save_index(dir.path(), None, &tables).unwrap();
        let (head, loaded) = load_index(dir.path());
        assert!(head.is_none());
        assert_eq!(loaded["inbox"].len(), 1);
        assert_eq!(loaded["inbox"].get_entry(b"mykey").unwrap().metadata.snippet, "test");
    }

    #[test]
    fn load_missing_index_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let (head, tables) = load_index(dir.path());
        assert!(head.is_none());
        assert!(tables.is_empty());
    }

    #[test]
    fn load_corrupt_index_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("index.json"), b"not json{{{").unwrap();
        let (head, tables) = load_index(dir.path());
        assert!(head.is_none());
        assert!(tables.is_empty());
    }

    #[test]
    fn truncate_snippet_short() {
        assert_eq!(truncate_snippet("hello"), "hello");
    }

    #[test]
    fn truncate_snippet_long() {
        let long = "a".repeat(300);
        let truncated = truncate_snippet(&long);
        assert!(truncated.len() <= MAX_SNIPPET_BYTES);
    }

    #[test]
    fn truncate_snippet_utf8_boundary() {
        // 'é' is 2 bytes in UTF-8; fill to boundary edge
        let s = "é".repeat(150); // 300 bytes
        let truncated = truncate_snippet(&s);
        assert!(truncated.len() <= MAX_SNIPPET_BYTES);
        assert!(truncated.is_char_boundary(truncated.len()));
    }
}
```

- [ ] **Step 2: Add hex serde helpers for Option<ContentId> to types.rs**

Add to the bottom of `crates/harmony-db/src/types.rs` (before the closing of the file):

```rust
/// Serde helper: serialize Option<ContentId> as nullable hex string.
pub(crate) fn ser_opt_cid<S: serde::Serializer>(
    cid: &Option<ContentId>,
    s: S,
) -> Result<S::Ok, S::Error> {
    match cid {
        Some(c) => s.serialize_str(&hex::encode(c.to_bytes())),
        None => s.serialize_none(),
    }
}

/// Serde helper: deserialize Option<ContentId> from nullable hex string.
pub(crate) fn de_opt_cid<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<ContentId>, D::Error> {
    let opt: Option<String> = Option::deserialize(d)?;
    match opt {
        None => Ok(None),
        Some(s) => {
            let bytes: [u8; 32] = hex::decode(&s)
                .map_err(serde::de::Error::custom)?
                .try_into()
                .map_err(|_| serde::de::Error::custom("expected 64 hex chars"))?;
            Ok(Some(ContentId::from_bytes(bytes)))
        }
    }
}
```

- [ ] **Step 3: Add module to lib.rs**

Update `crates/harmony-db/src/lib.rs`:

```rust
//! Content-addressed key-value database with named tables, atomic commits,
//! history diffing, and portable index rebuild from CAS.

mod error;
mod persist;
mod table;
mod types;

pub use error::DbError;
pub use types::{Entry, EntryMeta};
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p harmony-db -- persist
```

Expected: all 8 persist tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-db/src/persist.rs crates/harmony-db/src/types.rs crates/harmony-db/src/lib.rs
git commit -m "feat(harmony-db): atomic persistence with index and blob I/O (ZEB-97)"
```

---

### Task 5: HarmonyDb — Open, Insert, Get, Range, Remove

**Files:**
- Create: `crates/harmony-db/src/db.rs`
- Modify: `crates/harmony-db/src/lib.rs` (add module + re-exports)

The public-facing struct that orchestrates tables + persistence.

- [ ] **Step 1: Write the HarmonyDb implementation**

Create `crates/harmony-db/src/db.rs`:

```rust
use crate::error::DbError;
use crate::persist;
use crate::table::Table;
use crate::types::{Entry, EntryMeta};
use harmony_content::{ContentFlags, ContentId};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A content-addressed key-value database with named tables.
pub struct HarmonyDb {
    data_dir: PathBuf,
    tables: HashMap<String, Table>,
    head: Option<ContentId>,
}

impl HarmonyDb {
    /// Open or create a database at `data_dir`.
    /// Loads the local index cache if present; otherwise starts empty.
    pub fn open(data_dir: &Path) -> Result<Self, DbError> {
        persist::ensure_dirs(data_dir)?;
        let (head, tables) = persist::load_index(data_dir);
        Ok(HarmonyDb {
            data_dir: data_dir.to_path_buf(),
            tables,
            head,
        })
    }

    /// List all table names.
    pub fn table_names(&self) -> Vec<&str> {
        self.tables.keys().map(|s| s.as_str()).collect()
    }

    /// Number of entries in a table (0 if table doesn't exist).
    pub fn table_len(&self, table: &str) -> usize {
        self.tables.get(table).map_or(0, |t| t.len())
    }

    /// Insert a key-value pair into a table. Creates the table if it doesn't
    /// exist. Computes BLAKE3 CID for the value, stores the blob, upserts
    /// the entry. Returns the value's ContentId.
    pub fn insert(
        &mut self,
        table: &str,
        key: &[u8],
        value: &[u8],
        meta: EntryMeta,
    ) -> Result<ContentId, DbError> {
        let cid = persist::write_blob(&self.data_dir, value)?;
        let entry = Entry {
            key: key.to_vec(),
            value_cid: cid,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            metadata: EntryMeta {
                flags: meta.flags,
                snippet: persist::truncate_snippet(&meta.snippet),
            },
        };
        self.tables.entry(table.to_string()).or_insert_with(Table::new).upsert(entry);
        persist::save_index(&self.data_dir, self.head, &self.tables)?;
        Ok(cid)
    }

    /// Get the value bytes for a key in a table.
    pub fn get(&self, table: &str, key: &[u8]) -> Result<Option<Vec<u8>>, DbError> {
        let entry = match self.tables.get(table).and_then(|t| t.get_entry(key)) {
            Some(e) => e,
            None => return Ok(None),
        };
        persist::read_blob(&self.data_dir, &entry.value_cid)
    }

    /// Get just the entry (metadata + CID) without reading the blob.
    pub fn get_entry(&self, table: &str, key: &[u8]) -> Option<&Entry> {
        self.tables.get(table)?.get_entry(key)
    }

    /// Ordered slice over a key range [start, end) in a table.
    pub fn range(&self, table: &str, start: &[u8], end: &[u8]) -> &[Entry] {
        match self.tables.get(table) {
            Some(t) => t.range(start, end),
            None => &[],
        }
    }

    /// All entries in a table, sorted by key.
    pub fn entries(&self, table: &str) -> &[Entry] {
        match self.tables.get(table) {
            Some(t) => t.entries(),
            None => &[],
        }
    }

    /// Remove an entry by key from a table. Does NOT delete the value blob.
    pub fn remove(&mut self, table: &str, key: &[u8]) -> Result<Option<Entry>, DbError> {
        let removed = self.tables.get_mut(table).and_then(|t| t.remove(key));
        if removed.is_some() {
            persist::save_index(&self.data_dir, self.head, &self.tables)?;
        }
        Ok(removed)
    }

    /// Update metadata for an entry without touching the value.
    pub fn update_meta(
        &mut self,
        table: &str,
        key: &[u8],
        meta: EntryMeta,
    ) -> Result<(), DbError> {
        let found = self.tables.get_mut(table).map_or(false, |t| t.update_meta(key, meta));
        if !found {
            return Err(DbError::TableNotFound { name: table.to_string() });
        }
        persist::save_index(&self.data_dir, self.head, &self.tables)?;
        Ok(())
    }

    /// Current head commit CID (None if no commits yet).
    pub fn head(&self) -> Option<ContentId> {
        self.head
    }

    // -- Internal accessors used by commit module --

    pub(crate) fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    pub(crate) fn tables(&self) -> &HashMap<String, Table> {
        &self.tables
    }

    pub(crate) fn set_head(&mut self, head: Option<ContentId>) {
        self.head = head;
    }

    pub(crate) fn replace_tables(&mut self, tables: HashMap<String, Table>) {
        self.tables = tables;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(flags: u64, snippet: &str) -> EntryMeta {
        EntryMeta { flags, snippet: snippet.to_string() }
    }

    #[test]
    fn open_empty() {
        let dir = tempfile::tempdir().unwrap();
        let db = HarmonyDb::open(dir.path()).unwrap();
        assert!(db.table_names().is_empty());
        assert!(db.head().is_none());
    }

    #[test]
    fn insert_and_get() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        let cid = db.insert("inbox", b"msg1", b"hello world", meta(0, "greet")).unwrap();
        let value = db.get("inbox", b"msg1").unwrap().unwrap();
        assert_eq!(value, b"hello world");
        assert_eq!(db.get_entry("inbox", b"msg1").unwrap().value_cid, cid);
    }

    #[test]
    fn insert_upsert() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("t", b"k", b"old", meta(0, "")).unwrap();
        db.insert("t", b"k", b"new", meta(1, "")).unwrap();
        assert_eq!(db.table_len("t"), 1);
        assert_eq!(db.get("t", b"k").unwrap().unwrap(), b"new");
        assert_eq!(db.get_entry("t", b"k").unwrap().metadata.flags, 1);
    }

    #[test]
    fn get_nonexistent_table() {
        let dir = tempfile::tempdir().unwrap();
        let db = HarmonyDb::open(dir.path()).unwrap();
        assert!(db.get("nope", b"k").unwrap().is_none());
    }

    #[test]
    fn remove_entry() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("t", b"k", b"val", meta(0, "")).unwrap();
        let removed = db.remove("t", b"k").unwrap();
        assert!(removed.is_some());
        assert!(db.get("t", b"k").unwrap().is_none());
    }

    #[test]
    fn update_meta() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("t", b"k", b"val", meta(0, "old")).unwrap();
        db.update_meta("t", b"k", meta(1, "new")).unwrap();
        let e = db.get_entry("t", b"k").unwrap();
        assert_eq!(e.metadata.flags, 1);
        assert_eq!(e.metadata.snippet, "new");
    }

    #[test]
    fn range_query() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("t", b"a", b"1", meta(0, "")).unwrap();
        db.insert("t", b"b", b"2", meta(0, "")).unwrap();
        db.insert("t", b"c", b"3", meta(0, "")).unwrap();
        db.insert("t", b"d", b"4", meta(0, "")).unwrap();
        let range = db.range("t", b"b", b"d");
        assert_eq!(range.len(), 2);
        assert_eq!(range[0].key, b"b");
        assert_eq!(range[1].key, b"c");
    }

    #[test]
    fn multi_table() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("inbox", b"k", b"inbox_val", meta(0, "")).unwrap();
        db.insert("sent", b"k", b"sent_val", meta(0, "")).unwrap();
        assert_eq!(db.get("inbox", b"k").unwrap().unwrap(), b"inbox_val");
        assert_eq!(db.get("sent", b"k").unwrap().unwrap(), b"sent_val");
        assert_eq!(db.table_len("inbox"), 1);
        assert_eq!(db.table_len("sent"), 1);
    }

    #[test]
    fn open_existing_persists() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut db = HarmonyDb::open(dir.path()).unwrap();
            db.insert("t", b"k", b"val", meta(0, "hi")).unwrap();
        }
        let db = HarmonyDb::open(dir.path()).unwrap();
        assert_eq!(db.get_entry("t", b"k").unwrap().metadata.snippet, "hi");
        assert_eq!(db.get("t", b"k").unwrap().unwrap(), b"val");
    }

    #[test]
    fn snippet_truncated_on_insert() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        let long = "a".repeat(500);
        db.insert("t", b"k", b"v", meta(0, &long)).unwrap();
        let e = db.get_entry("t", b"k").unwrap();
        assert!(e.metadata.snippet.len() <= 256);
    }

    #[test]
    fn blob_dedup_across_tables() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        let cid1 = db.insert("inbox", b"k1", b"same_value", meta(0, "")).unwrap();
        let cid2 = db.insert("sent", b"k2", b"same_value", meta(0, "")).unwrap();
        assert_eq!(cid1, cid2);
        // Only one blob file
        let blob_count = std::fs::read_dir(dir.path().join("blobs"))
            .unwrap()
            .filter(|e| e.as_ref().unwrap().path().extension() == Some("bin".as_ref()))
            .count();
        assert_eq!(blob_count, 1);
    }
}
```

- [ ] **Step 2: Update lib.rs with re-exports**

Update `crates/harmony-db/src/lib.rs`:

```rust
//! Content-addressed key-value database with named tables, atomic commits,
//! history diffing, and portable index rebuild from CAS.

mod db;
mod error;
mod persist;
mod table;
mod types;

pub use db::HarmonyDb;
pub use error::DbError;
pub use types::{Entry, EntryMeta};
```

- [ ] **Step 3: Run all tests**

```bash
cargo test -p harmony-db
```

Expected: all tests pass (table + persist + db tests).

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-db/src/db.rs crates/harmony-db/src/lib.rs
git commit -m "feat(harmony-db): HarmonyDb with insert/get/range/remove across named tables (ZEB-97)"
```

---

### Task 6: Commit — Serialize Tables to CAS Snapshots

**Files:**
- Create: `crates/harmony-db/src/commit.rs`
- Modify: `crates/harmony-db/src/db.rs` (add commit method)
- Modify: `crates/harmony-db/src/lib.rs` (add module)

- [ ] **Step 1: Write the commit module**

Create `crates/harmony-db/src/commit.rs`:

```rust
use crate::error::DbError;
use crate::persist;
use crate::table::Table;
use crate::types::Entry;
use harmony_content::{BookStore, ContentFlags, ContentId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

const COMMIT_VERSION: u32 = 1;

/// Root manifest stored in commits/{root_cid}.bin.
#[derive(Debug, Serialize, Deserialize)]
struct RootManifest {
    version: u32,
    #[serde(
        serialize_with = "crate::types::ser_opt_cid",
        deserialize_with = "crate::types::de_opt_cid"
    )]
    parent: Option<ContentId>,
    /// Map of table name → page CID (hex).
    tables: HashMap<String, String>,
}

/// Serialize all tables and produce a root commit CID.
pub(crate) fn create_commit(
    data_dir: &Path,
    head: Option<ContentId>,
    tables: &HashMap<String, Table>,
    store: Option<&mut dyn BookStore>,
) -> Result<ContentId, DbError> {
    let commits_dir = data_dir.join("commits");
    let mut table_cids: HashMap<String, String> = HashMap::new();

    // Serialize each table's entries as a page blob.
    for (name, table) in tables {
        let page_bytes = serde_json::to_vec(table.entries())
            .map_err(|e| DbError::Serialize(e.to_string()))?;
        let page_cid = ContentId::for_book(&page_bytes, ContentFlags::default())
            .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
        let page_hex = hex::encode(page_cid.to_bytes());

        // Write page blob to commits/ (idempotent).
        let page_path = commits_dir.join(format!("{page_hex}.bin"));
        if !page_path.exists() {
            let tmp = commits_dir.join(format!("{page_hex}.bin.tmp"));
            std::fs::write(&tmp, &page_bytes)?;
            std::fs::rename(&tmp, &page_path)?;
        }

        // Push to BookStore if provided.
        if let Some(ref mut s) = store {
            let _ = s.insert_with_flags(&page_bytes, ContentFlags::default());
        }

        table_cids.insert(name.clone(), page_hex);
    }

    // Build root manifest.
    let manifest = RootManifest {
        version: COMMIT_VERSION,
        parent: head,
        tables: table_cids,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| DbError::Serialize(e.to_string()))?;
    let root_cid = ContentId::for_book(&manifest_bytes, ContentFlags::default())
        .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
    let root_hex = hex::encode(root_cid.to_bytes());

    // Write root manifest to commits/.
    let root_path = commits_dir.join(format!("{root_hex}.bin"));
    if !root_path.exists() {
        let tmp = commits_dir.join(format!("{root_hex}.bin.tmp"));
        std::fs::write(&tmp, &manifest_bytes)?;
        std::fs::rename(&tmp, &root_path)?;
    }

    // Push manifest + value blobs to BookStore.
    if let Some(ref mut s) = store {
        let _ = s.insert_with_flags(&manifest_bytes, ContentFlags::default());
        // Push value blobs that the store doesn't already have.
        for table in tables.values() {
            for entry in table.entries() {
                if !s.contains(&entry.value_cid) {
                    if let Ok(Some(blob)) = persist::read_blob(data_dir, &entry.value_cid) {
                        s.store(entry.value_cid, blob);
                    }
                }
            }
        }
    }

    Ok(root_cid)
}

/// Load a root manifest from local commits/ or from a BookStore.
pub(crate) fn load_manifest(
    data_dir: &Path,
    root_cid: ContentId,
    store: Option<&dyn BookStore>,
) -> Result<RootManifest, DbError> {
    let root_hex = hex::encode(root_cid.to_bytes());
    let local_path = data_dir.join("commits").join(format!("{root_hex}.bin"));

    let bytes = if local_path.exists() {
        std::fs::read(&local_path)?
    } else if let Some(s) = store {
        s.get(&root_cid)
            .map(|b| b.to_vec())
            .ok_or_else(|| DbError::CommitNotFound { cid: root_hex.clone() })?
    } else {
        return Err(DbError::CommitNotFound { cid: root_hex });
    };

    serde_json::from_slice(&bytes).map_err(|e| DbError::CorruptIndex(e.to_string()))
}

/// Load a table's page blob from local commits/ or from a BookStore.
fn load_page(
    data_dir: &Path,
    page_hex: &str,
    store: Option<&dyn BookStore>,
) -> Result<Vec<Entry>, DbError> {
    let local_path = data_dir.join("commits").join(format!("{page_hex}.bin"));

    let bytes = if local_path.exists() {
        std::fs::read(&local_path)?
    } else if let Some(s) = store {
        let page_bytes: [u8; 32] = hex::decode(page_hex)
            .map_err(|e| DbError::CorruptIndex(e.to_string()))?
            .try_into()
            .map_err(|_| DbError::CorruptIndex("bad page CID length".into()))?;
        let page_cid = ContentId::from_bytes(page_bytes);
        s.get(&page_cid)
            .map(|b| b.to_vec())
            .ok_or_else(|| DbError::CommitNotFound { cid: page_hex.to_string() })?
    } else {
        return Err(DbError::CommitNotFound { cid: page_hex.to_string() });
    };

    serde_json::from_slice(&bytes).map_err(|e| DbError::CorruptIndex(e.to_string()))
}

/// Rebuild tables from a commit snapshot.
pub(crate) fn rebuild(
    data_dir: &Path,
    root_cid: ContentId,
    store: Option<&dyn BookStore>,
) -> Result<HashMap<String, Table>, DbError> {
    let manifest = load_manifest(data_dir, root_cid, store)?;
    let mut tables = HashMap::new();

    for (name, page_hex) in &manifest.tables {
        let entries = load_page(data_dir, page_hex, store)?;
        let mut table = Table::new();
        for entry in entries {
            // Try to fetch missing value blobs from BookStore.
            let cid_hex = hex::encode(entry.value_cid.to_bytes());
            let local_blob = data_dir.join("blobs").join(format!("{cid_hex}.bin"));
            if !local_blob.exists() {
                if let Some(s) = store {
                    if let Some(blob_data) = s.get(&entry.value_cid) {
                        let _ = persist::write_blob_raw(data_dir, &cid_hex, blob_data);
                    }
                }
                // If still missing, that's OK — entry is visible but get() returns None.
            }
            table.upsert(entry);
        }
        tables.insert(name.clone(), table);
    }

    Ok(tables)
}

/// Diff two commits. Returns added/removed/changed entries per table.
pub(crate) fn diff_commits(
    data_dir: &Path,
    old_cid: ContentId,
    new_cid: ContentId,
) -> Result<crate::types::Diff, DbError> {
    let old_manifest = load_manifest(data_dir, old_cid, None)?;
    let new_manifest = load_manifest(data_dir, new_cid, None)?;

    let mut tables = HashMap::new();

    // Tables in both manifests.
    for (name, new_page_hex) in &new_manifest.tables {
        if let Some(old_page_hex) = old_manifest.tables.get(name) {
            if old_page_hex == new_page_hex {
                // Page CID identical — no changes in this table.
                continue;
            }
            let old_entries = load_page(data_dir, old_page_hex, None)?;
            let new_entries = load_page(data_dir, new_page_hex, None)?;
            tables.insert(name.clone(), diff_entries(&old_entries, &new_entries));
        } else {
            // Table only in new — all entries are added.
            let new_entries = load_page(data_dir, new_page_hex, None)?;
            tables.insert(name.clone(), crate::types::TableDiff {
                added: new_entries,
                removed: Vec::new(),
                changed: Vec::new(),
            });
        }
    }

    // Tables only in old — all entries are removed.
    for (name, old_page_hex) in &old_manifest.tables {
        if !new_manifest.tables.contains_key(name) {
            let old_entries = load_page(data_dir, old_page_hex, None)?;
            tables.insert(name.clone(), crate::types::TableDiff {
                added: Vec::new(),
                removed: old_entries,
                changed: Vec::new(),
            });
        }
    }

    Ok(crate::types::Diff { tables })
}

/// Two-pointer merge over two sorted entry slices.
fn diff_entries(old: &[Entry], new: &[Entry]) -> crate::types::TableDiff {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();

    let mut oi = 0;
    let mut ni = 0;

    while oi < old.len() && ni < new.len() {
        match old[oi].key.cmp(&new[ni].key) {
            std::cmp::Ordering::Less => {
                removed.push(old[oi].clone());
                oi += 1;
            }
            std::cmp::Ordering::Greater => {
                added.push(new[ni].clone());
                ni += 1;
            }
            std::cmp::Ordering::Equal => {
                if old[oi].value_cid != new[ni].value_cid {
                    changed.push((old[oi].clone(), new[ni].clone()));
                }
                oi += 1;
                ni += 1;
            }
        }
    }
    while oi < old.len() {
        removed.push(old[oi].clone());
        oi += 1;
    }
    while ni < new.len() {
        added.push(new[ni].clone());
        ni += 1;
    }

    crate::types::TableDiff { added, removed, changed }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EntryMeta;

    fn meta(snippet: &str) -> EntryMeta {
        EntryMeta { flags: 0, snippet: snippet.to_string() }
    }

    #[test]
    fn commit_produces_cid() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let mut tables = HashMap::new();
        let mut t = Table::new();
        let cid = ContentId::for_book(b"val", ContentFlags::default()).unwrap();
        t.upsert(Entry {
            key: b"k".to_vec(), value_cid: cid, timestamp: 1, metadata: meta(""),
        });
        tables.insert("t".to_string(), t);
        let root = create_commit(dir.path(), None, &tables, None).unwrap();
        // Root CID is deterministic — commit same data twice → same CID.
        let root2 = create_commit(dir.path(), None, &tables, None).unwrap();
        assert_eq!(root, root2);
    }

    #[test]
    fn commit_with_bookstore() {
        use harmony_content::MemoryBookStore;
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tables = HashMap::new();
        let mut t = Table::new();
        let value = b"hello world";
        let value_cid = persist::write_blob(dir.path(), value).unwrap();
        t.upsert(Entry {
            key: b"k".to_vec(), value_cid, timestamp: 1, metadata: meta(""),
        });
        tables.insert("t".to_string(), t);

        let mut bs = MemoryBookStore::new();
        let root = create_commit(dir.path(), None, &tables, Some(&mut bs)).unwrap();

        // BookStore should have: root manifest + page blob + value blob.
        assert!(bs.contains(&root));
        assert!(bs.contains(&value_cid));
    }

    #[test]
    fn rebuild_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tables = HashMap::new();
        let mut t = Table::new();
        let value_cid = persist::write_blob(dir.path(), b"the value").unwrap();
        t.upsert(Entry {
            key: b"mykey".to_vec(), value_cid, timestamp: 42, metadata: meta("snap"),
        });
        tables.insert("inbox".to_string(), t);

        let root = create_commit(dir.path(), None, &tables, None).unwrap();
        let rebuilt = rebuild(dir.path(), root, None).unwrap();

        assert_eq!(rebuilt["inbox"].len(), 1);
        let e = rebuilt["inbox"].get_entry(b"mykey").unwrap();
        assert_eq!(e.metadata.snippet, "snap");
        assert_eq!(e.value_cid, value_cid);
    }

    #[test]
    fn rebuild_from_bookstore() {
        use harmony_content::MemoryBookStore;
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tables = HashMap::new();
        let mut t = Table::new();
        let value_cid = persist::write_blob(dir.path(), b"value").unwrap();
        t.upsert(Entry {
            key: b"k".to_vec(), value_cid, timestamp: 1, metadata: meta(""),
        });
        tables.insert("t".to_string(), t);

        let mut bs = MemoryBookStore::new();
        let root = create_commit(dir.path(), None, &tables, Some(&mut bs)).unwrap();

        // Delete local commits directory to force BookStore fallback.
        std::fs::remove_dir_all(dir.path().join("commits")).unwrap();
        std::fs::create_dir_all(dir.path().join("commits")).unwrap();

        let rebuilt = rebuild(dir.path(), root, Some(&bs)).unwrap();
        assert_eq!(rebuilt["t"].len(), 1);
    }

    #[test]
    fn rebuild_orphaned_blobs() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tables = HashMap::new();
        let mut t = Table::new();
        let value_cid = persist::write_blob(dir.path(), b"value").unwrap();
        t.upsert(Entry {
            key: b"k".to_vec(), value_cid, timestamp: 1, metadata: meta(""),
        });
        tables.insert("t".to_string(), t);

        let root = create_commit(dir.path(), None, &tables, None).unwrap();

        // Delete the value blob to simulate orphan.
        let cid_hex = hex::encode(value_cid.to_bytes());
        std::fs::remove_file(dir.path().join("blobs").join(format!("{cid_hex}.bin"))).unwrap();

        // Rebuild should succeed — entry is present but blob is missing.
        let rebuilt = rebuild(dir.path(), root, None).unwrap();
        assert_eq!(rebuilt["t"].len(), 1);
        // get() for the value would return None (blob missing).
    }

    #[test]
    fn diff_no_changes() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tables = HashMap::new();
        let mut t = Table::new();
        let cid = ContentId::for_book(b"v", ContentFlags::default()).unwrap();
        t.upsert(Entry { key: b"k".to_vec(), value_cid: cid, timestamp: 1, metadata: meta("") });
        tables.insert("t".to_string(), t);

        let root = create_commit(dir.path(), None, &tables, None).unwrap();
        let d = diff_commits(dir.path(), root, root).unwrap();
        assert!(d.tables.is_empty()); // Same page CIDs → skipped.
    }

    #[test]
    fn diff_additions() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let cid_a = ContentId::for_book(b"a", ContentFlags::default()).unwrap();
        let cid_b = ContentId::for_book(b"b", ContentFlags::default()).unwrap();

        let mut tables1 = HashMap::new();
        let mut t1 = Table::new();
        t1.upsert(Entry { key: b"k1".to_vec(), value_cid: cid_a, timestamp: 1, metadata: meta("") });
        tables1.insert("t".to_string(), t1);
        let root1 = create_commit(dir.path(), None, &tables1, None).unwrap();

        let mut tables2 = HashMap::new();
        let mut t2 = Table::new();
        t2.upsert(Entry { key: b"k1".to_vec(), value_cid: cid_a, timestamp: 1, metadata: meta("") });
        t2.upsert(Entry { key: b"k2".to_vec(), value_cid: cid_b, timestamp: 2, metadata: meta("") });
        tables2.insert("t".to_string(), t2);
        let root2 = create_commit(dir.path(), Some(root1), &tables2, None).unwrap();

        let d = diff_commits(dir.path(), root1, root2).unwrap();
        assert_eq!(d.tables["t"].added.len(), 1);
        assert_eq!(d.tables["t"].added[0].key, b"k2");
        assert!(d.tables["t"].removed.is_empty());
    }

    #[test]
    fn diff_removals() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let cid_a = ContentId::for_book(b"a", ContentFlags::default()).unwrap();
        let cid_b = ContentId::for_book(b"b", ContentFlags::default()).unwrap();

        let mut tables1 = HashMap::new();
        let mut t1 = Table::new();
        t1.upsert(Entry { key: b"k1".to_vec(), value_cid: cid_a, timestamp: 1, metadata: meta("") });
        t1.upsert(Entry { key: b"k2".to_vec(), value_cid: cid_b, timestamp: 2, metadata: meta("") });
        tables1.insert("t".to_string(), t1);
        let root1 = create_commit(dir.path(), None, &tables1, None).unwrap();

        let mut tables2 = HashMap::new();
        let mut t2 = Table::new();
        t2.upsert(Entry { key: b"k1".to_vec(), value_cid: cid_a, timestamp: 1, metadata: meta("") });
        tables2.insert("t".to_string(), t2);
        let root2 = create_commit(dir.path(), Some(root1), &tables2, None).unwrap();

        let d = diff_commits(dir.path(), root1, root2).unwrap();
        assert_eq!(d.tables["t"].removed.len(), 1);
        assert_eq!(d.tables["t"].removed[0].key, b"k2");
    }

    #[test]
    fn diff_changes() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let cid_old = ContentId::for_book(b"old", ContentFlags::default()).unwrap();
        let cid_new = ContentId::for_book(b"new", ContentFlags::default()).unwrap();

        let mut tables1 = HashMap::new();
        let mut t1 = Table::new();
        t1.upsert(Entry { key: b"k".to_vec(), value_cid: cid_old, timestamp: 1, metadata: meta("") });
        tables1.insert("t".to_string(), t1);
        let root1 = create_commit(dir.path(), None, &tables1, None).unwrap();

        let mut tables2 = HashMap::new();
        let mut t2 = Table::new();
        t2.upsert(Entry { key: b"k".to_vec(), value_cid: cid_new, timestamp: 2, metadata: meta("") });
        tables2.insert("t".to_string(), t2);
        let root2 = create_commit(dir.path(), Some(root1), &tables2, None).unwrap();

        let d = diff_commits(dir.path(), root1, root2).unwrap();
        assert_eq!(d.tables["t"].changed.len(), 1);
        assert_eq!(d.tables["t"].changed[0].0.value_cid, cid_old);
        assert_eq!(d.tables["t"].changed[0].1.value_cid, cid_new);
    }

    #[test]
    fn diff_table_skip() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let cid = ContentId::for_book(b"v", ContentFlags::default()).unwrap();

        let mut tables = HashMap::new();
        let mut t1 = Table::new();
        t1.upsert(Entry { key: b"k".to_vec(), value_cid: cid, timestamp: 1, metadata: meta("") });
        tables.insert("unchanged".to_string(), t1);

        let mut t2 = Table::new();
        t2.upsert(Entry { key: b"x".to_vec(), value_cid: cid, timestamp: 1, metadata: meta("") });
        tables.insert("changed".to_string(), t2);

        let root1 = create_commit(dir.path(), None, &tables, None).unwrap();

        // Modify only "changed" table.
        let cid2 = ContentId::for_book(b"v2", ContentFlags::default()).unwrap();
        tables.get_mut("changed").unwrap().upsert(Entry {
            key: b"y".to_vec(), value_cid: cid2, timestamp: 2, metadata: meta(""),
        });
        let root2 = create_commit(dir.path(), Some(root1), &tables, None).unwrap();

        let d = diff_commits(dir.path(), root1, root2).unwrap();
        // "unchanged" table should not appear in diff (page CID identical).
        assert!(!d.tables.contains_key("unchanged"));
        assert!(d.tables.contains_key("changed"));
    }
}
```

- [ ] **Step 2: Add `write_blob_raw` helper to persist.rs**

Add to `crates/harmony-db/src/persist.rs`:

```rust
/// Write raw bytes to a blob file by hex CID (used during rebuild).
pub(crate) fn write_blob_raw(data_dir: &Path, cid_hex: &str, data: &[u8]) -> Result<(), DbError> {
    let blob_path = data_dir.join("blobs").join(format!("{cid_hex}.bin"));
    if blob_path.exists() {
        return Ok(());
    }
    let tmp = data_dir.join("blobs").join(format!("{cid_hex}.bin.tmp"));
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, &blob_path)?;
    Ok(())
}
```

- [ ] **Step 3: Add Diff types to types.rs**

Add to the public types in `crates/harmony-db/src/types.rs`:

```rust
use std::collections::HashMap;

/// Difference between two commits.
#[derive(Debug, Clone)]
pub struct Diff {
    /// Per-table differences. Tables with no changes are omitted.
    pub tables: HashMap<String, TableDiff>,
}

/// Differences within a single table between two commits.
#[derive(Debug, Clone)]
pub struct TableDiff {
    /// Entries present in new but not old.
    pub added: Vec<Entry>,
    /// Entries present in old but not new.
    pub removed: Vec<Entry>,
    /// Entries with same key but different value_cid: (old, new).
    pub changed: Vec<(Entry, Entry)>,
}
```

- [ ] **Step 4: Wire commit/diff/rebuild into HarmonyDb**

Add to `crates/harmony-db/src/db.rs`:

```rust
use crate::commit;

// Add these methods to the HarmonyDb impl block:

    /// Atomic snapshot: serialize all tables to CAS blobs, produce a root
    /// manifest, persist locally. Optionally pushes to a BookStore.
    pub fn commit(
        &mut self,
        store: Option<&mut dyn BookStore>,
    ) -> Result<ContentId, DbError> {
        let root = commit::create_commit(&self.data_dir, self.head, &self.tables, store)?;
        self.head = Some(root);
        persist::save_index(&self.data_dir, self.head, &self.tables)?;
        Ok(root)
    }

    /// Diff two commits. Returns entries added, removed, or changed per table.
    pub fn diff(&self, old: ContentId, new: ContentId) -> Result<crate::types::Diff, DbError> {
        commit::diff_commits(&self.data_dir, old, new)
    }

    /// Rebuild the in-memory index from a CAS commit snapshot.
    /// Replaces current state. Fetches locally first, falls back to BookStore.
    pub fn rebuild_from(
        &mut self,
        root: ContentId,
        store: Option<&dyn BookStore>,
    ) -> Result<(), DbError> {
        let tables = commit::rebuild(&self.data_dir, root, store)?;
        self.tables = tables;
        self.head = Some(root);
        persist::save_index(&self.data_dir, self.head, &self.tables)?;
        Ok(())
    }

    /// Open, then immediately rebuild from a CAS commit.
    pub fn open_from_cas(
        data_dir: &Path,
        root: ContentId,
        store: &dyn BookStore,
    ) -> Result<Self, DbError> {
        let mut db = Self::open(data_dir)?;
        db.rebuild_from(root, Some(store))?;
        Ok(db)
    }
```

Add the necessary import to the top of `db.rs`:

```rust
use harmony_content::{BookStore, ContentFlags, ContentId};
```

- [ ] **Step 5: Update lib.rs re-exports**

Update `crates/harmony-db/src/lib.rs`:

```rust
//! Content-addressed key-value database with named tables, atomic commits,
//! history diffing, and portable index rebuild from CAS.

mod commit;
mod db;
mod error;
mod persist;
mod table;
mod types;

pub use db::HarmonyDb;
pub use error::DbError;
pub use types::{Diff, Entry, EntryMeta, TableDiff};
```

- [ ] **Step 6: Run all tests**

```bash
cargo test -p harmony-db
```

Expected: all tests pass (~35 tests: 6 error + 10 table + 8 persist + ~11 commit + ~10 db).

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-db/
git commit -m "feat(harmony-db): commit snapshots, diff, and rebuild from CAS (ZEB-97)"
```

---

### Task 7: Integration Example — Mail Workflow

**Files:**
- Create: `crates/harmony-db/examples/mail_workflow.rs`

A runnable example that mimics the mail.rs lifecycle, validating fitness for the primary consumer.

- [ ] **Step 1: Write the example**

Create `crates/harmony-db/examples/mail_workflow.rs`:

```rust
//! Example: mail-like workflow using harmony-db.
//!
//! Demonstrates: multi-table DB, insert, mark-read via update_meta,
//! move-to-trash via remove+insert, commit, diff, rebuild.
//!
//! Run: cargo run -p harmony-db --example mail_workflow

use harmony_content::MemoryBookStore;
use harmony_db::{EntryMeta, HarmonyDb};

fn meta(read: bool, snippet: &str) -> EntryMeta {
    EntryMeta {
        flags: if read { 1 } else { 0 },
        snippet: snippet.to_string(),
    }
}

fn main() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = HarmonyDb::open(dir.path()).unwrap();

    println!("=== Receive 5 messages ===");
    for i in 0..5 {
        let key = format!("msg{i:04}");
        let body = format!("Message body #{i}");
        let snippet = format!("Subject {i}");
        db.insert("inbox", key.as_bytes(), body.as_bytes(), meta(false, &snippet))
            .unwrap();
    }
    println!("Inbox: {} messages", db.table_len("inbox"));

    // Initial commit.
    let mut store = MemoryBookStore::new();
    let root1 = db.commit(Some(&mut store)).unwrap();
    println!("Commit 1: {}", hex::encode(root1.to_bytes()));

    println!("\n=== Mark 2 as read ===");
    db.update_meta("inbox", b"msg0000", meta(true, "Subject 0")).unwrap();
    db.update_meta("inbox", b"msg0001", meta(true, "Subject 1")).unwrap();

    println!("\n=== Move msg0002 to trash ===");
    let entry = db.remove("inbox", b"msg0002").unwrap().unwrap();
    let body = db.get("inbox", b"msg0002"); // Already removed from index
    db.insert("trash", b"msg0002", b"Message body #2", meta(false, "Subject 2"))
        .unwrap();
    println!("Inbox: {}, Trash: {}", db.table_len("inbox"), db.table_len("trash"));

    println!("\n=== Send a message ===");
    db.insert("sent", b"out0001", b"Hey there!", meta(true, "Outgoing"))
        .unwrap();

    // Second commit.
    let root2 = db.commit(Some(&mut store)).unwrap();
    println!("Commit 2: {}", hex::encode(root2.to_bytes()));

    println!("\n=== Diff commit 1 vs 2 ===");
    let diff = db.diff(root1, root2).unwrap();
    for (table, td) in &diff.tables {
        println!(
            "  {table}: +{} added, -{} removed, ~{} changed",
            td.added.len(),
            td.removed.len(),
            td.changed.len()
        );
    }

    println!("\n=== Rebuild from CAS ===");
    let dir2 = tempfile::tempdir().unwrap();
    let db2 = HarmonyDb::open_from_cas(dir2.path(), root2, &store).unwrap();
    println!("Rebuilt inbox: {} messages", db2.table_len("inbox"));
    println!("Rebuilt sent: {} messages", db2.table_len("sent"));
    println!("Rebuilt trash: {} messages", db2.table_len("trash"));

    // Verify round-trip.
    assert_eq!(db2.table_len("inbox"), 4); // 5 - 1 moved to trash
    assert_eq!(db2.table_len("sent"), 1);
    assert_eq!(db2.table_len("trash"), 1);
    assert_eq!(
        db2.get_entry("inbox", b"msg0000").unwrap().metadata.flags,
        1, // marked read
    );
    assert!(db2.get("sent", b"out0001").unwrap().is_some());

    println!("\nAll assertions passed!");
}
```

- [ ] **Step 2: Run the example**

```bash
cargo run -p harmony-db --example mail_workflow
```

Expected: prints the workflow steps and "All assertions passed!" at the end.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-db/examples/mail_workflow.rs
git commit -m "feat(harmony-db): mail workflow integration example (ZEB-97)"
```

---

### Task 8: Final Verification and Cleanup

- [ ] **Step 1: Run full test suite**

```bash
cargo test -p harmony-db
```

Expected: all tests pass.

- [ ] **Step 2: Run clippy**

```bash
cargo clippy -p harmony-db -- -D warnings
```

Expected: no warnings.

- [ ] **Step 3: Verify the example runs**

```bash
cargo run -p harmony-db --example mail_workflow
```

Expected: passes.

- [ ] **Step 4: Check the crate builds in the workspace context**

```bash
cargo check --workspace
```

Expected: full workspace compiles. harmony-db doesn't break other crates.

- [ ] **Step 5: Commit any clippy/cleanup fixes**

If clippy found issues, fix and commit:

```bash
git add -A crates/harmony-db/
git commit -m "fix(harmony-db): address clippy warnings (ZEB-97)"
```

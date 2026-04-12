use crate::commit;
use crate::error::DbError;
use crate::persist;
use crate::table::Table;
use crate::types::{Entry, EntryMeta};
use harmony_content::book::BookStore;
use harmony_content::ContentId;
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
    pub fn open(data_dir: &Path) -> Result<Self, DbError> {
        persist::ensure_dirs(data_dir)?;
        let (head, tables) = persist::load_index(data_dir);
        Ok(HarmonyDb {
            data_dir: data_dir.to_path_buf(),
            tables,
            head,
        })
    }

    pub fn table_names(&self) -> Vec<&str> {
        self.tables.keys().map(|s| s.as_str()).collect()
    }

    pub fn table_len(&self, table: &str) -> usize {
        self.tables.get(table).map_or(0, |t| t.len())
    }

    /// Insert a key-value pair into a table. Creates the table if needed.
    /// Computes BLAKE3 CID for the value, stores blob, upserts entry.
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
        self.tables
            .entry(table.to_string())
            .or_insert_with(Table::new)
            .upsert(entry);
        persist::save_index(&self.data_dir, self.head, &self.tables)?;
        Ok(cid)
    }

    /// Get the value bytes for a key.
    pub fn get(&self, table: &str, key: &[u8]) -> Result<Option<Vec<u8>>, DbError> {
        let entry = match self.tables.get(table).and_then(|t| t.get_entry(key)) {
            Some(e) => e,
            None => return Ok(None),
        };
        persist::read_blob(&self.data_dir, &entry.value_cid)
    }

    /// Get just the entry without reading the blob.
    pub fn get_entry(&self, table: &str, key: &[u8]) -> Option<&Entry> {
        self.tables.get(table)?.get_entry(key)
    }

    /// Ordered slice over a key range [start, end).
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

    /// Remove an entry by key. Does NOT delete the value blob.
    pub fn remove(&mut self, table: &str, key: &[u8]) -> Result<Option<Entry>, DbError> {
        let removed = self.tables.get_mut(table).and_then(|t| t.remove(key));
        if removed.is_some() {
            persist::save_index(&self.data_dir, self.head, &self.tables)?;
        }
        Ok(removed)
    }

    /// Update metadata for an entry.
    pub fn update_meta(
        &mut self,
        table: &str,
        key: &[u8],
        meta: EntryMeta,
    ) -> Result<(), DbError> {
        let truncated_meta = EntryMeta {
            flags: meta.flags,
            snippet: persist::truncate_snippet(&meta.snippet),
        };
        let t = self.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound {
            name: table.to_string(),
        })?;
        if !t.update_meta(key, truncated_meta) {
            return Err(DbError::EntryNotFound { table: table.to_string() });
        }
        persist::save_index(&self.data_dir, self.head, &self.tables)?;
        Ok(())
    }

    /// Atomic snapshot: serialize all tables to CAS blobs, produce root manifest.
    pub fn commit(
        &mut self,
        store: Option<&mut dyn BookStore>,
    ) -> Result<ContentId, DbError> {
        let root = commit::create_commit(&self.data_dir, self.head, &self.tables, store)?;
        self.head = Some(root);
        persist::save_index(&self.data_dir, self.head, &self.tables)?;
        Ok(root)
    }

    /// Diff two commits.
    pub fn diff(&self, old: ContentId, new: ContentId) -> Result<crate::types::Diff, DbError> {
        commit::diff_commits(&self.data_dir, old, new)
    }

    /// Rebuild in-memory index from a CAS commit snapshot.
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

    /// Current head commit CID.
    pub fn head(&self) -> Option<ContentId> {
        self.head
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(flags: u64, snippet: &str) -> EntryMeta {
        EntryMeta {
            flags,
            snippet: snippet.to_string(),
        }
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
        let cid = db
            .insert("inbox", b"msg1", b"hello world", meta(0, "greet"))
            .unwrap();
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
    fn update_meta_test() {
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
        db.insert("inbox", b"k", b"inbox_val", meta(0, ""))
            .unwrap();
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
    fn update_meta_truncates_snippet() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("t", b"k", b"v", meta(0, "short")).unwrap();
        let long = "x".repeat(500);
        db.update_meta("t", b"k", meta(1, &long)).unwrap();
        let e = db.get_entry("t", b"k").unwrap();
        assert!(e.metadata.snippet.len() <= 256);
    }

    #[test]
    fn blob_dedup_across_tables() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        let cid1 = db
            .insert("inbox", b"k1", b"same_value", meta(0, ""))
            .unwrap();
        let cid2 = db
            .insert("sent", b"k2", b"same_value", meta(0, ""))
            .unwrap();
        assert_eq!(cid1, cid2);
        let blob_count = std::fs::read_dir(dir.path().join("blobs"))
            .unwrap()
            .filter(|e| e.as_ref().unwrap().path().extension() == Some("bin".as_ref()))
            .count();
        assert_eq!(blob_count, 1);
    }
}

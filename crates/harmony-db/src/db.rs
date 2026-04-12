use crate::error::DbError;
use crate::persist;
use crate::prolly::node::Node;
use crate::prolly::ProllyTree;
use crate::types::{Entry, EntryMeta};
use harmony_content::book::BookStore;
use harmony_content::{ContentFlags, ContentId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

const COMMIT_VERSION: u32 = 2;

#[derive(Debug, Serialize, Deserialize)]
struct CommitManifest {
    version: u32,
    #[serde(
        serialize_with = "crate::types::ser_opt_cid",
        deserialize_with = "crate::types::de_opt_cid"
    )]
    parent: Option<ContentId>,
    tables: BTreeMap<String, String>, // name → root CID hex
}

/// A content-addressed key-value database with named tables.
pub struct HarmonyDb {
    data_dir: PathBuf,
    tables: HashMap<String, ProllyTree>,
    head: Option<ContentId>,
}

impl HarmonyDb {
    /// Open or create a database at `data_dir`.
    pub fn open(data_dir: &Path) -> Result<Self, DbError> {
        persist::ensure_dirs(data_dir)?;
        let (head, table_roots) = persist::load_roots(data_dir);
        let mut tables = HashMap::new();
        for (name, root_cid) in table_roots {
            let tree = ProllyTree::from_root(root_cid, data_dir)?;
            tables.insert(name, tree);
        }
        Ok(HarmonyDb {
            data_dir: data_dir.to_path_buf(),
            tables,
            head,
        })
    }

    pub fn table_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tables.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
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
            .or_insert_with(ProllyTree::new)
            .insert(entry, &self.data_dir)?;
        self.save_roots()?;
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
    /// Removes the table from the index when it becomes empty.
    pub fn remove(&mut self, table: &str, key: &[u8]) -> Result<Option<Entry>, DbError> {
        let (removed, new_root) = match self.tables.get_mut(table) {
            Some(t) => t.remove(key, &self.data_dir)?,
            None => (None, None),
        };
        if removed.is_some() && new_root.is_none() {
            self.tables.remove(table);
        }
        if removed.is_some() {
            self.save_roots()?;
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
        let truncated_snippet = persist::truncate_snippet(&meta.snippet);
        let t = self
            .tables
            .get_mut(table)
            .ok_or_else(|| DbError::TableNotFound {
                name: table.to_string(),
            })?;
        if !t.update_meta(key, meta.flags, truncated_snippet, &self.data_dir)? {
            return Err(DbError::EntryNotFound {
                table: table.to_string(),
            });
        }
        self.save_roots()?;
        Ok(())
    }

    /// Atomic snapshot: serialize all table trees to CAS, produce root manifest.
    pub fn commit(
        &mut self,
        mut store: Option<&mut dyn BookStore>,
    ) -> Result<ContentId, DbError> {
        let commits_dir = self.data_dir.join("commits");
        let mut table_cids: BTreeMap<String, String> = BTreeMap::new();

        for (name, tree) in &self.tables {
            if let Some(root_cid) = tree.root() {
                table_cids.insert(name.clone(), hex::encode(root_cid.to_bytes()));
            }
        }

        let manifest = CommitManifest {
            version: COMMIT_VERSION,
            parent: self.head,
            tables: table_cids,
        };
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| DbError::Serialize(e.to_string()))?;
        let root_cid = ContentId::for_book(&manifest_bytes, ContentFlags::default())
            .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
        let root_hex = hex::encode(root_cid.to_bytes());

        let root_path = commits_dir.join(format!("{root_hex}.bin"));
        if !root_path.exists() {
            let tmp = commits_dir.join(format!("{root_hex}.bin.tmp"));
            std::fs::write(&tmp, &manifest_bytes)?;
            std::fs::rename(&tmp, &root_path)?;
        }

        // Push to BookStore if provided.
        if let Some(ref mut s) = store {
            // Push tree nodes: for each table, walk the tree and push all nodes.
            for tree in self.tables.values() {
                if let Some(root) = tree.root() {
                    push_tree_nodes(&self.data_dir, root, *s)?;
                }
            }

            // Push value blobs referenced by leaf entries.
            for tree in self.tables.values() {
                for entry in tree.entries() {
                    if !s.contains(&entry.value_cid) {
                        if let Some(blob) = persist::read_blob(&self.data_dir, &entry.value_cid)? {
                            s.store(entry.value_cid, blob);
                        }
                    }
                }
            }

            // Manifest last — atomic visibility.
            let stored_manifest_cid = s
                .insert_with_flags(&manifest_bytes, ContentFlags::default())
                .map_err(|e| {
                    DbError::Serialize(format!("BookStore manifest push failed: {e:?}"))
                })?;
            if stored_manifest_cid != root_cid {
                return Err(DbError::CorruptIndex(format!(
                    "manifest CID mismatch: local {} vs store {}",
                    hex::encode(root_cid.to_bytes()),
                    hex::encode(stored_manifest_cid.to_bytes())
                )));
            }
        }

        // Persist before updating in-memory state so a failed save
        // doesn't leave head inconsistent with disk.
        let new_head = Some(root_cid);
        let table_roots: BTreeMap<String, Option<ContentId>> = self
            .tables
            .iter()
            .map(|(n, t): (&String, &ProllyTree)| (n.clone(), t.root()))
            .collect();
        persist::save_roots(&self.data_dir, new_head, &table_roots)?;
        self.head = new_head;
        Ok(root_cid)
    }

    /// Diff two commits. Optionally provide a BookStore to fetch
    /// commit blobs not cached locally.
    pub fn diff(
        &self,
        old: ContentId,
        new: ContentId,
        store: Option<&dyn BookStore>,
    ) -> Result<crate::types::Diff, DbError> {
        let old_manifest = load_manifest(&self.data_dir, old, store)?;
        let new_manifest = load_manifest(&self.data_dir, new, store)?;

        let mut tables = HashMap::new();

        for (name, new_root_hex) in &new_manifest.tables {
            let new_root_cid = hex_to_cid(new_root_hex)?;
            if let Some(old_root_hex) = old_manifest.tables.get(name) {
                if old_root_hex == new_root_hex {
                    continue; // Skip identical tables.
                }
                let old_root_cid = hex_to_cid(old_root_hex)?;
                // Prefetch nodes from store if needed.
                if let Some(s) = store {
                    prefetch_tree_nodes(&self.data_dir, old_root_cid, s)?;
                    prefetch_tree_nodes(&self.data_dir, new_root_cid, s)?;
                }
                let td = crate::prolly::diff::diff_trees(
                    &self.data_dir,
                    Some(old_root_cid),
                    Some(new_root_cid),
                )?;
                if !td.added.is_empty() || !td.removed.is_empty() || !td.changed.is_empty() {
                    tables.insert(name.clone(), td);
                }
            } else {
                // New table — all entries are additions.
                if let Some(s) = store {
                    prefetch_tree_nodes(&self.data_dir, new_root_cid, s)?;
                }
                let td = crate::prolly::diff::diff_trees(
                    &self.data_dir,
                    None,
                    Some(new_root_cid),
                )?;
                tables.insert(name.clone(), td);
            }
        }

        for (name, old_root_hex) in &old_manifest.tables {
            if !new_manifest.tables.contains_key(name) {
                let old_root_cid = hex_to_cid(old_root_hex)?;
                if let Some(s) = store {
                    prefetch_tree_nodes(&self.data_dir, old_root_cid, s)?;
                }
                let td = crate::prolly::diff::diff_trees(
                    &self.data_dir,
                    Some(old_root_cid),
                    None,
                )?;
                tables.insert(name.clone(), td);
            }
        }

        Ok(crate::types::Diff { tables })
    }

    /// Rebuild in-memory index from a CAS commit snapshot.
    pub fn rebuild_from(
        &mut self,
        root: ContentId,
        store: Option<&dyn BookStore>,
    ) -> Result<(), DbError> {
        let manifest = load_manifest(&self.data_dir, root, store)?;
        let mut new_tables = HashMap::new();

        for (name, root_hex) in &manifest.tables {
            let root_cid = hex_to_cid(root_hex)?;

            // Prefetch tree nodes from store to local commits/.
            if let Some(s) = store {
                prefetch_tree_nodes(&self.data_dir, root_cid, s)?;
            }

            let tree = ProllyTree::from_root(root_cid, &self.data_dir)?;

            // Prefetch value blobs.
            for entry in tree.entries() {
                let cid_hex = hex::encode(entry.value_cid.to_bytes());
                let local_blob = self.data_dir.join("blobs").join(format!("{cid_hex}.bin"));
                if !local_blob.exists() {
                    if let Some(s) = store {
                        if let Some(blob_data) = s.get(&entry.value_cid) {
                            persist::write_blob_raw(&self.data_dir, &cid_hex, blob_data)?;
                        }
                    }
                }
            }

            new_tables.insert(name.clone(), tree);
        }

        let new_head = Some(root);
        self.tables = new_tables;
        self.head = new_head;
        self.save_roots()?;
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

    /// Persist current roots (head + per-table root CIDs) to index.json.
    fn save_roots(&self) -> Result<(), DbError> {
        let table_roots: BTreeMap<String, Option<ContentId>> = self
            .tables
            .iter()
            .map(|(name, tree)| (name.clone(), tree.root()))
            .collect();
        persist::save_roots(&self.data_dir, self.head, &table_roots)
    }
}

// ---- Helpers ----

fn hex_to_cid(hex_str: &str) -> Result<ContentId, DbError> {
    let bytes: [u8; 32] = hex::decode(hex_str)
        .map_err(|e| DbError::CorruptIndex(e.to_string()))?
        .try_into()
        .map_err(|_| DbError::CorruptIndex("bad CID length".into()))?;
    Ok(ContentId::from_bytes(bytes))
}

fn load_manifest(
    data_dir: &Path,
    root_cid: ContentId,
    store: Option<&dyn BookStore>,
) -> Result<CommitManifest, DbError> {
    let root_hex = hex::encode(root_cid.to_bytes());
    let local_path = data_dir.join("commits").join(format!("{root_hex}.bin"));

    let (bytes, from_store) = if local_path.exists() {
        (std::fs::read(&local_path)?, false)
    } else if let Some(s) = store {
        let fetched = s
            .get(&root_cid)
            .map(|b| b.to_vec())
            .ok_or_else(|| DbError::CommitNotFound {
                cid: root_hex.clone(),
            })?;
        (fetched, true)
    } else {
        return Err(DbError::CommitNotFound { cid: root_hex });
    };

    // Verify content hashes to expected CID for both local and remote reads.
    let computed = ContentId::for_book(&bytes, ContentFlags::default())
        .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
    if computed != root_cid {
        return Err(DbError::CorruptIndex(format!(
            "manifest content mismatch for {root_hex}"
        )));
    }

    let manifest: CommitManifest = serde_json::from_slice(&bytes)
        .map_err(|e| DbError::CorruptIndex(e.to_string()))?;
    if manifest.version != COMMIT_VERSION {
        return Err(DbError::CorruptIndex(format!(
            "unsupported commit version: {} (expected {})",
            manifest.version, COMMIT_VERSION
        )));
    }

    // Cache locally after validation.
    if from_store {
        let tmp = local_path.with_extension("bin.tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &local_path)?;
    }

    Ok(manifest)
}

/// Recursively push all tree nodes to a BookStore.
/// Pushes children before parents so data is available before references.
fn push_tree_nodes(
    data_dir: &Path,
    cid: ContentId,
    store: &mut dyn BookStore,
) -> Result<(), DbError> {
    if store.contains(&cid) {
        return Ok(());
    }
    let hex = hex::encode(cid.to_bytes());
    let path = data_dir.join("commits").join(format!("{hex}.bin"));
    let bytes = std::fs::read(&path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            DbError::CommitNotFound { cid: hex.clone() }
        } else {
            DbError::Io(e)
        }
    })?;

    // Parse node to discover children.
    let node: Node =
        postcard::from_bytes(&bytes).map_err(|e| DbError::CorruptIndex(e.to_string()))?;
    match &node {
        Node::Branch(children) => {
            // Push children first.
            for child in children {
                push_tree_nodes(data_dir, ContentId::from_bytes(child.child_cid), store)?;
            }
        }
        Node::Leaf(_) => {} // No children to push.
    }

    // Push this node and verify the stored CID matches.
    let stored_cid = store
        .insert_with_flags(&bytes, ContentFlags::default())
        .map_err(|e| DbError::Serialize(format!("BookStore node push failed: {e:?}")))?;
    if stored_cid != cid {
        return Err(DbError::CorruptIndex(format!(
            "local node content mismatch for {hex}: stored as {}",
            hex::encode(stored_cid.to_bytes())
        )));
    }
    Ok(())
}

/// Recursively prefetch tree nodes from a BookStore to local commits/.
fn prefetch_tree_nodes(
    data_dir: &Path,
    cid: ContentId,
    store: &dyn BookStore,
) -> Result<(), DbError> {
    let hex = hex::encode(cid.to_bytes());
    let local_path = data_dir.join("commits").join(format!("{hex}.bin"));
    if local_path.exists() {
        // Already cached — but still need to recurse into children
        // that might not be cached.
        let bytes = std::fs::read(&local_path)?;
        if let Ok(Node::Branch(children)) = postcard::from_bytes::<Node>(&bytes) {
            for child in children {
                prefetch_tree_nodes(
                    data_dir,
                    ContentId::from_bytes(child.child_cid),
                    store,
                )?;
            }
        }
        return Ok(());
    }

    // Fetch from store.
    let bytes = match store.get(&cid) {
        Some(b) => b.to_vec(),
        None => return Ok(()), // Best-effort.
    };

    // Verify CID.
    let computed = ContentId::for_book(&bytes, ContentFlags::default())
        .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
    if computed != cid {
        return Err(DbError::CorruptIndex(format!(
            "tree node content mismatch for {hex}"
        )));
    }

    // Cache locally.
    let tmp = local_path.with_extension("bin.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, &local_path)?;

    // Recurse into children.
    if let Ok(Node::Branch(children)) = postcard::from_bytes::<Node>(&bytes) {
        for child in children {
            prefetch_tree_nodes(data_dir, ContentId::from_bytes(child.child_cid), store)?;
        }
    }

    Ok(())
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

    #[test]
    fn commit_and_diff() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();

        db.insert("t", b"k1", b"v1", meta(0, "")).unwrap();
        let root1 = db.commit(None).unwrap();

        db.insert("t", b"k2", b"v2", meta(0, "")).unwrap();
        let root2 = db.commit(None).unwrap();

        let diff = db.diff(root1, root2, None).unwrap();
        assert_eq!(diff.tables["t"].added.len(), 1);
        assert_eq!(diff.tables["t"].added[0].key, b"k2");
        assert!(diff.tables["t"].removed.is_empty());
    }

    #[test]
    fn commit_with_bookstore_and_rebuild() {
        use harmony_content::book::MemoryBookStore;

        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();

        db.insert("inbox", b"m1", b"hello", meta(0, "greet"))
            .unwrap();
        db.insert("inbox", b"m2", b"world", meta(1, "planet"))
            .unwrap();

        let mut store = MemoryBookStore::new();
        let root = db.commit(Some(&mut store)).unwrap();

        // Rebuild in a fresh directory from BookStore only.
        let dir2 = tempfile::tempdir().unwrap();
        let db2 = HarmonyDb::open_from_cas(dir2.path(), root, &store).unwrap();
        assert_eq!(db2.table_len("inbox"), 2);
        assert_eq!(
            db2.get("inbox", b"m1").unwrap().unwrap(),
            b"hello"
        );
        assert_eq!(
            db2.get_entry("inbox", b"m2").unwrap().metadata.flags,
            1
        );
    }
}

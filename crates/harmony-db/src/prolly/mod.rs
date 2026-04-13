pub(crate) mod chunker;
pub(crate) mod diff;
pub(crate) mod mutate;
pub(crate) mod node;

use crate::error::DbError;
use crate::types::Entry;
use chunker::{chunk_items, ChunkerConfig};
use harmony_content::ContentId;
use node::{BranchEntry, LeafEntry, Node};
use std::path::Path;

#[derive(Debug, Clone)]
pub(crate) struct ProllyTree {
    root: Option<ContentId>,
    cache: Vec<Entry>,
    config: ChunkerConfig,
}

impl ProllyTree {
    pub fn new() -> Self {
        ProllyTree {
            root: None,
            cache: Vec::new(),
            config: ChunkerConfig::default_4k(),
        }
    }

    /// Load tree from existing root CID — walks tree to populate cache.
    pub fn from_root(root_cid: ContentId, data_dir: &Path) -> Result<Self, DbError> {
        let mut entries = Vec::new();
        collect_entries(data_dir, root_cid, &mut entries)?;
        entries.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(ProllyTree {
            root: Some(root_cid),
            cache: entries,
            config: ChunkerConfig::default_4k(),
        })
    }

    pub fn root(&self) -> Option<ContentId> {
        self.root
    }

    // --- Reads (from cache, no disk I/O) ---

    pub fn get_entry(&self, key: &[u8]) -> Option<&Entry> {
        let idx = self
            .cache
            .binary_search_by(|e| e.key.as_slice().cmp(key))
            .ok()?;
        Some(&self.cache[idx])
    }

    pub fn range(&self, start: &[u8], end: &[u8]) -> &[Entry] {
        let lo = match self.cache.binary_search_by(|e| e.key.as_slice().cmp(start)) {
            Ok(i) | Err(i) => i,
        };
        let hi = match self.cache.binary_search_by(|e| e.key.as_slice().cmp(end)) {
            Ok(i) | Err(i) => i,
        };
        if lo >= hi {
            return &[];
        }
        &self.cache[lo..hi]
    }

    pub fn entries(&self) -> &[Entry] {
        &self.cache
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    // --- Writes (update cache + rebuild tree) ---

    pub fn insert(
        &mut self,
        entry: Entry,
        data_dir: &Path,
    ) -> Result<Option<ContentId>, DbError> {
        let old_cache = self.cache.clone();
        let old_root = self.root;
        match self.cache.binary_search_by(|e| e.key.cmp(&entry.key)) {
            Ok(idx) => self.cache[idx] = entry.clone(),
            Err(idx) => self.cache.insert(idx, entry.clone()),
        }
        let leaf_entry = LeafEntry::from_entry(&entry);
        match mutate::incremental_insert(data_dir, self.root, &leaf_entry, &self.config) {
            Ok(new_root) => {
                self.root = new_root;
                Ok(new_root)
            }
            Err(e) => {
                self.cache = old_cache;
                self.root = old_root;
                Err(e)
            }
        }
    }

    pub fn remove(
        &mut self,
        key: &[u8],
        data_dir: &Path,
    ) -> Result<(Option<Entry>, Option<ContentId>), DbError> {
        let old_cache = self.cache.clone();
        let old_root = self.root;
        let removed = match self.cache.binary_search_by(|e| e.key.as_slice().cmp(key)) {
            Ok(idx) => Some(self.cache.remove(idx)),
            Err(_) => None,
        };
        let new_root = if removed.is_some() {
            match self.rebuild_tree(data_dir) {
                Ok(root) => root,
                Err(e) => { self.cache = old_cache; self.root = old_root; return Err(e); }
            }
        } else {
            self.root
        };
        Ok((removed, new_root))
    }

    pub fn update_meta(
        &mut self,
        key: &[u8],
        flags: u64,
        snippet: String,
        data_dir: &Path,
    ) -> Result<bool, DbError> {
        let idx = match self.cache.binary_search_by(|e| e.key.as_slice().cmp(key)) {
            Ok(i) => i,
            Err(_) => return Ok(false),
        };
        let old_cache = self.cache.clone();
        let old_root = self.root;
        self.cache[idx].metadata.flags = flags;
        self.cache[idx].metadata.snippet = snippet;
        match self.rebuild_tree(data_dir) {
            Ok(_) => Ok(true),
            Err(e) => { self.cache = old_cache; self.root = old_root; Err(e) }
        }
    }

    fn rebuild_tree(&mut self, data_dir: &Path) -> Result<Option<ContentId>, DbError> {
        self.root = build_tree(&self.cache, &self.config, data_dir)?;
        Ok(self.root)
    }
}

/// Build a Prolly Tree bottom-up from sorted entries. Returns root CID.
pub(crate) fn build_tree(
    entries: &[Entry],
    config: &ChunkerConfig,
    data_dir: &Path,
) -> Result<Option<ContentId>, DbError> {
    if entries.is_empty() {
        return Ok(None);
    }

    let leaf_entries: Vec<LeafEntry> = entries.iter().map(LeafEntry::from_entry).collect();

    // Chunk into leaf nodes.
    let leaf_chunks = chunk_items(
        &leaf_entries,
        config,
        |e| e.approx_size(),
        |e| e.key.as_slice(),
    );

    // Write leaf nodes, collect branch entries.
    let mut branch_entries: Vec<BranchEntry> = Vec::with_capacity(leaf_chunks.len());
    for chunk in &leaf_chunks {
        let boundary_key = chunk.last().unwrap().key.clone();
        let node = Node::Leaf(chunk.clone());
        let cid = node.write_to_cas(data_dir)?;
        branch_entries.push(BranchEntry {
            boundary_key,
            child_cid: cid.to_bytes(),
        });
    }

    if branch_entries.len() == 1 {
        return Ok(Some(ContentId::from_bytes(branch_entries[0].child_cid)));
    }

    // Build branch levels until single root.
    loop {
        let prev_len = branch_entries.len();
        let branch_chunks = chunk_items(
            &branch_entries,
            config,
            |e| e.approx_size(),
            |e| e.boundary_key.as_slice(),
        );
        let mut next_level: Vec<BranchEntry> = Vec::with_capacity(branch_chunks.len());
        for chunk in &branch_chunks {
            let boundary_key = chunk.last().unwrap().boundary_key.clone();
            let node = Node::Branch(chunk.clone());
            let cid = node.write_to_cas(data_dir)?;
            next_level.push(BranchEntry {
                boundary_key,
                child_cid: cid.to_bytes(),
            });
        }
        if next_level.len() == 1 {
            return Ok(Some(ContentId::from_bytes(next_level[0].child_cid)));
        }
        // Guard against infinite loop: if chunking didn't reduce the count
        // (e.g., large boundary keys where each entry exceeds min_size),
        // force convergence by wrapping everything into a single root.
        if next_level.len() >= prev_len {
            let node = Node::Branch(next_level);
            let cid = node.write_to_cas(data_dir)?;
            return Ok(Some(cid));
        }
        branch_entries = next_level;
    }
}

/// Walk tree recursively, collect all leaf entries as public Entry type.
fn collect_entries(
    data_dir: &Path,
    cid: ContentId,
    out: &mut Vec<Entry>,
) -> Result<(), DbError> {
    let node = Node::read_from_cas(data_dir, cid)?;
    match node {
        Node::Leaf(entries) => {
            for e in entries {
                out.push(e.to_entry());
            }
        }
        Node::Branch(children) => {
            for child in children {
                collect_entries(data_dir, ContentId::from_bytes(child.child_cid), out)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist;
    use crate::types::EntryMeta;

    fn make_entry(key: &[u8], value: &[u8], data_dir: &Path) -> Entry {
        let cid = persist::write_blob(data_dir, value).unwrap();
        Entry {
            key: key.to_vec(),
            value_cid: cid,
            timestamp: 1000,
            metadata: EntryMeta {
                flags: 0,
                snippet: "test".to_string(),
            },
        }
    }

    #[test]
    fn empty_tree() {
        let tree = ProllyTree::new();
        assert!(tree.root().is_none());
        assert_eq!(tree.len(), 0);
        assert!(tree.get_entry(b"anything").is_none());
        assert!(tree.range(b"a", b"z").is_empty());
    }

    #[test]
    fn insert_and_get() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tree = ProllyTree::new();
        let entry = make_entry(b"hello", b"world", dir.path());
        tree.insert(entry.clone(), dir.path()).unwrap();

        assert_eq!(tree.len(), 1);
        assert!(tree.root().is_some());
        let found = tree.get_entry(b"hello").unwrap();
        assert_eq!(found.key, b"hello");
        assert_eq!(found.value_cid, entry.value_cid);
    }

    #[test]
    fn insert_upsert() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tree = ProllyTree::new();
        let e1 = make_entry(b"key", b"value1", dir.path());
        let e2 = make_entry(b"key", b"value2", dir.path());

        tree.insert(e1, dir.path()).unwrap();
        tree.insert(e2.clone(), dir.path()).unwrap();

        assert_eq!(tree.len(), 1);
        let found = tree.get_entry(b"key").unwrap();
        assert_eq!(found.value_cid, e2.value_cid);
    }

    #[test]
    fn insert_sorted_order() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tree = ProllyTree::new();
        tree.insert(make_entry(b"cherry", b"c", dir.path()), dir.path())
            .unwrap();
        tree.insert(make_entry(b"apple", b"a", dir.path()), dir.path())
            .unwrap();
        tree.insert(make_entry(b"banana", b"b", dir.path()), dir.path())
            .unwrap();

        let keys: Vec<&[u8]> = tree.entries().iter().map(|e| e.key.as_slice()).collect();
        assert_eq!(keys, vec![b"apple".as_slice(), b"banana", b"cherry"]);
    }

    #[test]
    fn range_query() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tree = ProllyTree::new();
        for k in &[b"a", b"b", b"c", b"d"] {
            tree.insert(make_entry(*k, *k, dir.path()), dir.path())
                .unwrap();
        }

        let result = tree.range(b"b", b"d");
        let keys: Vec<&[u8]> = result.iter().map(|e| e.key.as_slice()).collect();
        assert_eq!(keys, vec![b"b".as_slice(), b"c"]);
    }

    #[test]
    fn remove_entry() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tree = ProllyTree::new();
        tree.insert(make_entry(b"a", b"va", dir.path()), dir.path())
            .unwrap();
        tree.insert(make_entry(b"b", b"vb", dir.path()), dir.path())
            .unwrap();

        let (removed, _) = tree.remove(b"a", dir.path()).unwrap();
        assert!(removed.is_some());
        assert!(tree.get_entry(b"a").is_none());
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn update_meta_no_rechunk() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tree = ProllyTree::new();
        tree.insert(make_entry(b"key", b"val", dir.path()), dir.path())
            .unwrap();
        let root_before = tree.root();

        let updated = tree
            .update_meta(b"key", 42, "new snippet".to_string(), dir.path())
            .unwrap();
        assert!(updated);

        let found = tree.get_entry(b"key").unwrap();
        assert_eq!(found.metadata.flags, 42);
        assert_eq!(found.metadata.snippet, "new snippet");

        // Root should change because metadata changed the node content.
        assert_ne!(tree.root(), root_before);
    }

    #[test]
    fn large_insert_builds_tree() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tree = ProllyTree::new();
        for i in 0..1000 {
            let key = format!("entry-{i:04}").into_bytes();
            let val = format!("value-{i:04}").into_bytes();
            tree.insert(make_entry(&key, &val, dir.path()), dir.path())
                .unwrap();
        }

        assert_eq!(tree.len(), 1000);

        // Verify all entries are retrievable.
        for i in 0..1000 {
            let key = format!("entry-{i:04}").into_bytes();
            assert!(
                tree.get_entry(&key).is_some(),
                "entry-{i:04} should be found"
            );
        }

        // Root should be a Branch node (not a single Leaf).
        let root_cid = tree.root().unwrap();
        let root_node = Node::read_from_cas(dir.path(), root_cid).unwrap();
        assert!(
            matches!(root_node, Node::Branch(_)),
            "root of 1000-entry tree should be a Branch node"
        );
    }

    #[test]
    fn history_independence() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir1.path()).unwrap();
        persist::ensure_dirs(dir2.path()).unwrap();

        let entries: Vec<(Vec<u8>, Vec<u8>)> = (0..100)
            .map(|i| {
                let key = format!("entry-{i:04}").into_bytes();
                let val = format!("value-{i:04}").into_bytes();
                (key, val)
            })
            .collect();

        // Tree 1: insert in ascending order.
        let mut tree1 = ProllyTree::new();
        for (key, val) in entries.iter() {
            tree1
                .insert(make_entry(key, val, dir1.path()), dir1.path())
                .unwrap();
        }

        // Tree 2: insert in reverse order.
        let mut tree2 = ProllyTree::new();
        for (key, val) in entries.iter().rev() {
            tree2
                .insert(make_entry(key, val, dir2.path()), dir2.path())
                .unwrap();
        }

        assert_eq!(
            tree1.root(),
            tree2.root(),
            "trees built from same data in different order must have the same root CID"
        );
    }

    #[test]
    fn value_update_stability() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tree = ProllyTree::new();
        for i in 0..100 {
            let key = format!("entry-{i:04}").into_bytes();
            let val = format!("value-{i:04}").into_bytes();
            tree.insert(make_entry(&key, &val, dir.path()), dir.path())
                .unwrap();
        }

        let root_before = tree.root();

        // Update one entry with a different value.
        let updated_entry = make_entry(b"entry-0050", b"new-value", dir.path());
        tree.insert(updated_entry, dir.path()).unwrap();

        assert_ne!(
            tree.root(),
            root_before,
            "changing a value should produce a different root"
        );
    }

    #[test]
    fn from_root_rebuilds_cache() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let mut tree = ProllyTree::new();
        for i in 0..50 {
            let key = format!("entry-{i:04}").into_bytes();
            let val = format!("value-{i:04}").into_bytes();
            tree.insert(make_entry(&key, &val, dir.path()), dir.path())
                .unwrap();
        }

        let root_cid = tree.root().unwrap();

        // Rebuild tree from root.
        let rebuilt = ProllyTree::from_root(root_cid, dir.path()).unwrap();

        assert_eq!(rebuilt.len(), tree.len());
        assert_eq!(rebuilt.root(), tree.root());

        // Verify all entries match.
        for original in tree.entries() {
            let found = rebuilt.get_entry(&original.key).unwrap();
            assert_eq!(found, original);
        }
    }
}

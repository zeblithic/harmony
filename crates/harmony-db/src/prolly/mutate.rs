// Incremental O(log n) Prolly Tree mutations.
//
// Instead of rebuilding the entire tree from scratch on every mutation,
// walk from root to the affected leaf, mutate, rechunk the affected
// region until boundaries re-converge, and cascade new CIDs up to root.

use crate::error::DbError;
use harmony_content::ContentId;
use std::path::Path;
use super::chunker::{chunk_items, ChunkerConfig};
use super::node::{BranchEntry, LeafEntry, Node};

/// One level of the descent path from root to leaf.
#[derive(Debug, Clone)]
struct PathLevel {
    /// The branch entries at this level (full node contents).
    entries: Vec<BranchEntry>,
    /// Which child index we followed during descent.
    child_idx: usize,
}

/// Complete path from root to a leaf, plus the leaf's entries.
#[derive(Debug, Clone)]
struct TreePath {
    /// Branch levels from root (index 0) to parent-of-leaf (last).
    levels: Vec<PathLevel>,
    /// The leaf entries at the bottom of the path.
    leaf_entries: Vec<LeafEntry>,
}

/// Result of rechunking a leaf region after mutation.
struct RechunkResult {
    /// New leaf children (boundary_key + CID) to replace in the parent branch.
    new_children: Vec<BranchEntry>,
    /// How many additional sibling leaves were absorbed during convergence
    /// (0 = only the target leaf was rechunked).
    #[allow(dead_code)]
    siblings_consumed: usize,
}

/// Walk from root to the leaf whose key range contains `key`.
///
/// Uses boundary_key (highest key in each subtree) to route: find the first
/// branch entry where boundary_key >= key. If key exceeds all boundary keys,
/// use the last child.
fn walk_to_leaf(
    data_dir: &Path,
    root: ContentId,
    key: &[u8],
) -> Result<TreePath, DbError> {
    let mut levels = Vec::new();
    let mut cid = root;

    loop {
        let node = Node::read_from_cas(data_dir, cid)?;
        match node {
            Node::Leaf(entries) => {
                return Ok(TreePath { levels, leaf_entries: entries });
            }
            Node::Branch(entries) => {
                let child_idx = entries
                    .iter()
                    .position(|e| e.boundary_key.as_slice() >= key)
                    .unwrap_or(entries.len() - 1);
                cid = ContentId::from_bytes(entries[child_idx].child_cid);
                levels.push(PathLevel { entries, child_idx });
            }
        }
    }
}

/// Load entries from a leaf node. Errors if the CID points to a branch.
fn load_leaf_entries(data_dir: &Path, cid: ContentId) -> Result<Vec<LeafEntry>, DbError> {
    let node = Node::read_from_cas(data_dir, cid)?;
    match node {
        Node::Leaf(entries) => Ok(entries),
        Node::Branch(_) => Err(DbError::CorruptIndex(
            "expected leaf node during rechunk".into(),
        )),
    }
}

/// Rechunk a mutated leaf's entries, loading sibling leaves if boundaries
/// shift (convergence check).
///
/// Processes `mutated_entries` through the CDF chunker. After processing all
/// entries, checks if the next sibling leaf's first entry would trigger a
/// boundary. If yes: converged (the rest of the tree is unchanged). If no:
/// absorbs the sibling's entries and continues until convergence.
fn rechunk_leaf(
    data_dir: &Path,
    path: &TreePath,
    mutated_entries: Vec<LeafEntry>,
    config: &ChunkerConfig,
) -> Result<RechunkResult, DbError> {
    let parent = path.levels.last();
    let start_idx = parent.map_or(0, |p| p.child_idx);

    let mut new_children: Vec<BranchEntry> = Vec::new();
    let mut current_chunk: Vec<LeafEntry> = Vec::new();
    let mut current_size: usize = 0;
    let mut siblings_consumed: usize = 0;

    // Process entries batch by batch. First batch = mutated leaf entries.
    // Subsequent batches = absorbed sibling entries.
    let mut entries_to_process = mutated_entries;

    loop {
        // Run entries through the CDF chunker.
        for entry in &entries_to_process {
            let entry_size = entry.approx_size();
            let key = &entry.key;

            if !current_chunk.is_empty()
                && config.is_boundary(current_size, entry_size, key)
            {
                // Boundary fired — finalize current chunk as a new leaf.
                let bk = current_chunk.last().unwrap().key.clone();
                let node = Node::Leaf(std::mem::take(&mut current_chunk));
                let cid = node.write_to_cas(data_dir)?;
                new_children.push(BranchEntry {
                    boundary_key: bk,
                    child_cid: cid.to_bytes(),
                });
                current_size = 0;
            }

            current_chunk.push(entry.clone());
            current_size += entry_size;
        }

        // All entries processed. Check convergence with next sibling.
        if let Some(parent) = parent {
            let next_idx = start_idx + 1 + siblings_consumed;
            if next_idx < parent.entries.len() {
                let sibling_cid =
                    ContentId::from_bytes(parent.entries[next_idx].child_cid);
                let sibling_entries = load_leaf_entries(data_dir, sibling_cid)?;

                if let Some(first) = sibling_entries.first() {
                    if !current_chunk.is_empty()
                        && config.is_boundary(
                            current_size,
                            first.approx_size(),
                            &first.key,
                        )
                    {
                        // Converged! Next sibling's first entry triggers a
                        // boundary, so the rest of the tree is unchanged.
                        let bk = current_chunk.last().unwrap().key.clone();
                        let node = Node::Leaf(std::mem::take(&mut current_chunk));
                        let cid = node.write_to_cas(data_dir)?;
                        new_children.push(BranchEntry {
                            boundary_key: bk,
                            child_cid: cid.to_bytes(),
                        });
                        break;
                    }

                    // Not converged — absorb this sibling and continue.
                    siblings_consumed += 1;
                    entries_to_process = sibling_entries;
                    continue;
                }
            }
        }

        // No more siblings or no parent. Finalize the tail chunk.
        if !current_chunk.is_empty() {
            let bk = current_chunk.last().unwrap().key.clone();
            let node = Node::Leaf(std::mem::take(&mut current_chunk));
            let cid = node.write_to_cas(data_dir)?;
            new_children.push(BranchEntry {
                boundary_key: bk,
                child_cid: cid.to_bytes(),
            });
        }
        break;
    }

    Ok(RechunkResult {
        new_children,
        siblings_consumed,
    })
}

/// Cascade new children up through branch levels to produce a new root CID.
///
/// At each level: splice new children into the branch entries, rechunk the
/// branch, and propagate upward. At the root, build additional branch levels
/// if needed (tree height may grow or shrink).
fn cascade_up(
    data_dir: &Path,
    path: &TreePath,
    new_leaf_children: Vec<BranchEntry>,
    siblings_consumed: usize,
    config: &ChunkerConfig,
) -> Result<Option<ContentId>, DbError> {
    if path.levels.is_empty() {
        // Root was a leaf — promote new children to root.
        return finish_to_root(data_dir, new_leaf_children, config);
    }

    let mut replacement = new_leaf_children;
    let mut replace_count = 1 + siblings_consumed;
    let num_levels = path.levels.len();

    for (i, level) in path.levels.iter().rev().enumerate() {
        let is_root_level = i == num_levels - 1;

        // Splice new children into this level's branch entries.
        let mut entries = level.entries.clone();
        let start = level.child_idx;
        let end = (start + replace_count).min(entries.len());
        entries.splice(start..end, replacement.into_iter());

        if entries.is_empty() {
            // Branch level emptied — propagate upward.
            replacement = Vec::new();
            replace_count = 1;
            continue;
        }

        if entries.len() == 1 && is_root_level {
            // Single child at root → child becomes the root.
            return Ok(Some(ContentId::from_bytes(entries[0].child_cid)));
        }

        if is_root_level {
            // At root level, write and potentially build more levels.
            return finish_to_root(data_dir, entries, config);
        }

        // Non-root level — rechunk the branch entries.
        let chunks = chunk_items(
            &entries,
            config,
            |e| e.approx_size(),
            |e| e.boundary_key.as_slice(),
        );
        replacement = Vec::new();
        for chunk in &chunks {
            let bk = chunk.last().unwrap().boundary_key.clone();
            let node = Node::Branch(chunk.clone());
            let cid = node.write_to_cas(data_dir)?;
            replacement.push(BranchEntry {
                boundary_key: bk,
                child_cid: cid.to_bytes(),
            });
        }
        replace_count = 1; // Higher levels: replace just this one branch.
    }

    // Should not reach here — loop returns at root level.
    finish_to_root(data_dir, replacement, config)
}

/// Build branch levels from entries until a single root is produced.
/// Same logic as the branch loop in `build_tree`.
fn finish_to_root(
    data_dir: &Path,
    branch_entries: Vec<BranchEntry>,
    config: &ChunkerConfig,
) -> Result<Option<ContentId>, DbError> {
    if branch_entries.is_empty() {
        return Ok(None);
    }
    if branch_entries.len() == 1 {
        return Ok(Some(ContentId::from_bytes(branch_entries[0].child_cid)));
    }

    let mut entries = branch_entries;
    loop {
        let prev_len = entries.len();
        let chunks = chunk_items(
            &entries,
            config,
            |e| e.approx_size(),
            |e| e.boundary_key.as_slice(),
        );
        let mut next: Vec<BranchEntry> = Vec::new();
        for chunk in &chunks {
            let bk = chunk.last().unwrap().boundary_key.clone();
            let node = Node::Branch(chunk.clone());
            let cid = node.write_to_cas(data_dir)?;
            next.push(BranchEntry {
                boundary_key: bk,
                child_cid: cid.to_bytes(),
            });
        }
        if next.len() == 1 {
            return Ok(Some(ContentId::from_bytes(next[0].child_cid)));
        }
        // Convergence guard: if chunking didn't reduce count, force single root.
        if next.len() >= prev_len {
            let node = Node::Branch(next);
            let cid = node.write_to_cas(data_dir)?;
            return Ok(Some(cid));
        }
        entries = next;
    }
}

/// Incrementally insert an entry into a Prolly Tree.
///
/// Walk to the target leaf, insert the entry (or upsert if key exists),
/// rechunk with boundary convergence, cascade up to a new root CID.
pub(crate) fn incremental_insert(
    data_dir: &Path,
    root: Option<ContentId>,
    entry: &LeafEntry,
    config: &ChunkerConfig,
) -> Result<Option<ContentId>, DbError> {
    match root {
        None => {
            // Empty tree — create a single leaf with this entry.
            let node = Node::Leaf(vec![entry.clone()]);
            let cid = node.write_to_cas(data_dir)?;
            Ok(Some(cid))
        }
        Some(root_cid) => {
            let path = walk_to_leaf(data_dir, root_cid, &entry.key)?;

            let mut leaf = path.leaf_entries.clone();
            match leaf.binary_search_by(|e| e.key.cmp(&entry.key)) {
                Ok(idx) => leaf[idx] = entry.clone(),   // upsert
                Err(idx) => leaf.insert(idx, entry.clone()), // insert
            }

            let rechunk = rechunk_leaf(data_dir, &path, leaf, config)?;
            cascade_up(
                data_dir,
                &path,
                rechunk.new_children,
                rechunk.siblings_consumed,
                config,
            )
        }
    }
}

/// Incrementally remove an entry from a Prolly Tree.
///
/// Walk to the leaf containing the key, remove the entry, rechunk with
/// boundary convergence, cascade up. Returns None if the tree becomes empty.
pub(crate) fn incremental_remove(
    data_dir: &Path,
    root: ContentId,
    key: &[u8],
    config: &ChunkerConfig,
) -> Result<Option<ContentId>, DbError> {
    let path = walk_to_leaf(data_dir, root, key)?;

    let mut leaf = path.leaf_entries.clone();
    match leaf.binary_search_by(|e| e.key.as_slice().cmp(key)) {
        Ok(idx) => {
            leaf.remove(idx);
        }
        Err(_) => {
            // Key not found in this leaf — tree unchanged.
            return Ok(Some(root));
        }
    }

    if leaf.is_empty() && path.levels.is_empty() {
        // Removed the last entry from a single-leaf tree.
        return Ok(None);
    }

    let rechunk = rechunk_leaf(data_dir, &path, leaf, config)?;
    cascade_up(
        data_dir,
        &path,
        rechunk.new_children,
        rechunk.siblings_consumed,
        config,
    )
}

/// Incrementally update metadata for an entry in a Prolly Tree.
///
/// Walk to the leaf containing the key, update flags and snippet, rechunk
/// (in case snippet size change affects boundaries), cascade up.
/// Returns None if the key was not found.
pub(crate) fn incremental_update_meta(
    data_dir: &Path,
    root: ContentId,
    key: &[u8],
    flags: u64,
    snippet: String,
    config: &ChunkerConfig,
) -> Result<Option<ContentId>, DbError> {
    let path = walk_to_leaf(data_dir, root, key)?;

    let mut leaf = path.leaf_entries.clone();
    let idx = match leaf.binary_search_by(|e| e.key.as_slice().cmp(key)) {
        Ok(i) => i,
        Err(_) => return Ok(None), // Key not found.
    };
    leaf[idx].flags = flags;
    leaf[idx].snippet = snippet;

    let rechunk = rechunk_leaf(data_dir, &path, leaf, config)?;
    cascade_up(
        data_dir,
        &path,
        rechunk.new_children,
        rechunk.siblings_consumed,
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist;
    use crate::prolly::build_tree;
    use crate::types::{Entry, EntryMeta};

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

    fn setup_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        dir
    }

    #[test]
    fn walk_to_leaf_finds_correct_leaf() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        // Build a tree with enough entries to have multiple leaves.
        let entries: Vec<Entry> = (0..200u32)
            .map(|i| make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir.path(),
            ))
            .collect();
        let root = build_tree(&entries, &config, dir.path()).unwrap().unwrap();

        // Walk to a key in the middle.
        let target_key = b"k-000100";
        let path = walk_to_leaf(dir.path(), root, target_key).unwrap();

        // Should have at least one branch level (200 entries > 1 leaf).
        assert!(!path.levels.is_empty(), "200-entry tree should have branch levels");

        // The leaf should contain the target key.
        let found = path.leaf_entries.iter().any(|e| e.key == target_key);
        assert!(found, "leaf should contain the target key k-000100");

        // Leaf entries should be sorted.
        for w in path.leaf_entries.windows(2) {
            assert!(w[0].key < w[1].key, "leaf entries should be sorted");
        }
    }

    #[test]
    fn rechunk_after_insert_produces_valid_leaves() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        let entries: Vec<Entry> = (0..200u32)
            .map(|i| make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir.path(),
            ))
            .collect();
        let root = build_tree(&entries, &config, dir.path()).unwrap().unwrap();

        // Walk to a leaf in the middle.
        let target_key = b"k-000100";
        let path = walk_to_leaf(dir.path(), root, target_key).unwrap();

        // Insert a new entry into the leaf.
        let new_entry_data = make_entry(b"k-000100-new", b"inserted", dir.path());
        let new_leaf_entry = LeafEntry::from_entry(&new_entry_data);
        let mut leaf = path.leaf_entries.clone();
        let ins_idx = leaf.binary_search_by(|e| e.key.as_slice().cmp(b"k-000100-new".as_slice()))
            .unwrap_err();
        leaf.insert(ins_idx, new_leaf_entry);

        // Rechunk.
        let result = rechunk_leaf(dir.path(), &path, leaf, &config).unwrap();

        // Should produce at least one child.
        assert!(!result.new_children.is_empty(), "rechunk should produce children");

        // All children should be valid leaf nodes.
        for child in &result.new_children {
            let node = Node::read_from_cas(dir.path(), ContentId::from_bytes(child.child_cid)).unwrap();
            assert!(matches!(node, Node::Leaf(_)), "child should be a leaf");
        }

        // Boundary keys should be sorted.
        for w in result.new_children.windows(2) {
            assert!(
                w[0].boundary_key < w[1].boundary_key,
                "boundary keys should be sorted"
            );
        }
    }

    #[test]
    fn incremental_insert_matches_rebuild() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        // Build initial tree with 200 entries.
        let mut entries: Vec<Entry> = (0..200u32)
            .map(|i| make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir.path(),
            ))
            .collect();
        let initial_root = build_tree(&entries, &config, dir.path()).unwrap();

        // Insert one new entry incrementally.
        let new_entry = make_entry(b"k-000100-new", b"inserted", dir.path());
        let new_leaf_entry = LeafEntry::from_entry(&new_entry);
        let incremental_root = incremental_insert(
            dir.path(),
            initial_root,
            &new_leaf_entry,
            &config,
        ).unwrap();

        // Build reference tree with all 201 entries from scratch.
        entries.push(new_entry);
        entries.sort_by(|a, b| a.key.cmp(&b.key));
        let reference_root = build_tree(&entries, &config, dir.path()).unwrap();

        assert_eq!(
            incremental_root, reference_root,
            "incremental insert must produce same root CID as full rebuild"
        );
    }

    #[test]
    fn incremental_insert_into_empty_tree() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        let entry = make_entry(b"first-key", b"first-value", dir.path());
        let leaf_entry = LeafEntry::from_entry(&entry);
        let root = incremental_insert(dir.path(), None, &leaf_entry, &config).unwrap();

        assert!(root.is_some(), "inserting into empty tree should produce a root");

        // Should be a single leaf node.
        let node = Node::read_from_cas(dir.path(), root.unwrap()).unwrap();
        match node {
            Node::Leaf(entries) => {
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].key, b"first-key");
            }
            _ => panic!("expected leaf node"),
        }
    }

    #[test]
    fn incremental_upsert_matches_rebuild() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        let mut entries: Vec<Entry> = (0..100u32)
            .map(|i| make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir.path(),
            ))
            .collect();
        let initial_root = build_tree(&entries, &config, dir.path()).unwrap();

        // Upsert: same key, different value.
        let updated_entry = make_entry(b"k-000050", b"updated-value", dir.path());
        let updated_leaf = LeafEntry::from_entry(&updated_entry);
        let incremental_root = incremental_insert(
            dir.path(),
            initial_root,
            &updated_leaf,
            &config,
        ).unwrap();

        // Build reference with updated entry.
        entries[50] = updated_entry;
        let reference_root = build_tree(&entries, &config, dir.path()).unwrap();

        assert_eq!(incremental_root, reference_root);
    }

    #[test]
    fn incremental_remove_matches_rebuild() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        let entries: Vec<Entry> = (0..200u32)
            .map(|i| make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir.path(),
            ))
            .collect();
        let initial_root = build_tree(&entries, &config, dir.path()).unwrap().unwrap();

        // Remove entry 100 incrementally.
        let remove_key = format!("k-{:06}", 100).into_bytes();
        let incremental_root = incremental_remove(
            dir.path(),
            initial_root,
            &remove_key,
            &config,
        ).unwrap();

        // Build reference tree without entry 100.
        let remaining: Vec<Entry> = entries.into_iter()
            .filter(|e| e.key != remove_key)
            .collect();
        let reference_root = build_tree(&remaining, &config, dir.path()).unwrap();

        assert_eq!(
            incremental_root, reference_root,
            "incremental remove must produce same root CID as full rebuild"
        );
    }

    #[test]
    fn incremental_remove_last_entry() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        let entry = make_entry(b"only-key", b"only-value", dir.path());
        let root = build_tree(&[entry], &config, dir.path()).unwrap().unwrap();

        let result = incremental_remove(dir.path(), root, b"only-key", &config).unwrap();
        assert!(result.is_none(), "removing last entry should produce None root");
    }

    #[test]
    fn incremental_remove_nonexistent_key() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        let entries: Vec<Entry> = (0..10u32)
            .map(|i| make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir.path(),
            ))
            .collect();
        let root = build_tree(&entries, &config, dir.path()).unwrap().unwrap();

        // Try to remove a key that doesn't exist.
        let result = incremental_remove(dir.path(), root, b"nonexistent", &config).unwrap();
        assert_eq!(result, Some(root), "removing nonexistent key should return same root");
    }

    #[test]
    fn incremental_update_meta_matches_rebuild() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        let mut entries: Vec<Entry> = (0..100u32)
            .map(|i| make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir.path(),
            ))
            .collect();
        let initial_root = build_tree(&entries, &config, dir.path()).unwrap().unwrap();

        // Update metadata on entry 50.
        let incremental_root = incremental_update_meta(
            dir.path(),
            initial_root,
            format!("k-{:06}", 50).as_bytes(),
            42,
            "updated snippet".to_string(),
            &config,
        ).unwrap();

        assert!(incremental_root.is_some(), "update_meta should return a root");

        // Build reference with the same metadata change.
        entries[50].metadata.flags = 42;
        entries[50].metadata.snippet = "updated snippet".to_string();
        let reference_root = build_tree(&entries, &config, dir.path()).unwrap();

        assert_eq!(
            incremental_root, reference_root,
            "incremental update_meta must produce same root CID as full rebuild"
        );
    }

    #[test]
    fn incremental_sequence_matches_rebuild() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        // Insert 100 entries one at a time using incremental_insert.
        let entries: Vec<Entry> = (0..100u32)
            .map(|i| make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir.path(),
            ))
            .collect();

        let mut root: Option<ContentId> = None;
        for entry in &entries {
            let leaf_entry = LeafEntry::from_entry(entry);
            root = incremental_insert(dir.path(), root, &leaf_entry, &config).unwrap();
        }

        // Compare with full rebuild.
        let reference = build_tree(&entries, &config, dir.path()).unwrap();
        assert_eq!(root, reference, "100 sequential incremental inserts must match full rebuild");
    }

    #[test]
    fn incremental_mixed_ops_matches_rebuild() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        // Start with 50 entries.
        let entries: Vec<Entry> = (0..50u32)
            .map(|i| make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir.path(),
            ))
            .collect();

        let mut root: Option<ContentId> = None;
        for entry in &entries {
            let leaf_entry = LeafEntry::from_entry(entry);
            root = incremental_insert(dir.path(), root, &leaf_entry, &config).unwrap();
        }

        // Remove entries 10, 20, 30.
        for i in [10u32, 20, 30] {
            let key = format!("k-{i:06}").into_bytes();
            root = incremental_remove(dir.path(), root.unwrap(), &key, &config).unwrap();
        }

        // Update metadata on entry 25.
        root = incremental_update_meta(
            dir.path(),
            root.unwrap(),
            format!("k-{:06}", 25).as_bytes(),
            99,
            "mixed-test".to_string(),
            &config,
        ).unwrap();

        // Insert 5 new entries.
        for i in 50..55u32 {
            let entry = make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir.path(),
            );
            let leaf_entry = LeafEntry::from_entry(&entry);
            root = incremental_insert(dir.path(), root, &leaf_entry, &config).unwrap();
        }

        // Build reference from the expected final state.
        let mut expected: Vec<Entry> = (0..55u32)
            .filter(|i| *i != 10 && *i != 20 && *i != 30)
            .map(|i| make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir.path(),
            ))
            .collect();
        // Apply metadata update to entry 25.
        if let Some(e) = expected.iter_mut().find(|e| e.key == format!("k-{:06}", 25).as_bytes()) {
            e.metadata.flags = 99;
            e.metadata.snippet = "mixed-test".to_string();
        }
        let reference = build_tree(&expected, &config, dir.path()).unwrap();

        assert_eq!(root, reference, "mixed operations must match full rebuild");
    }

    #[test]
    fn incremental_order_independence() {
        let dir1 = setup_dir();
        let dir2 = setup_dir();
        let config = ChunkerConfig::default_4k();

        let entries: Vec<Entry> = (0..100u32)
            .map(|i| make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir1.path(),
            ))
            .collect();

        // Insert in ascending order.
        let mut root1: Option<ContentId> = None;
        for entry in &entries {
            let leaf_entry = LeafEntry::from_entry(entry);
            root1 = incremental_insert(dir1.path(), root1, &leaf_entry, &config).unwrap();
        }

        // Insert in descending order (different dir to avoid CAS collisions).
        // Re-create entries with dir2 blobs.
        let entries2: Vec<Entry> = (0..100u32)
            .map(|i| make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir2.path(),
            ))
            .collect();
        let mut root2: Option<ContentId> = None;
        for entry in entries2.iter().rev() {
            let leaf_entry = LeafEntry::from_entry(entry);
            root2 = incremental_insert(dir2.path(), root2, &leaf_entry, &config).unwrap();
        }

        assert_eq!(
            root1, root2,
            "same entries inserted in different order must produce same root CID"
        );
    }

    #[test]
    fn incremental_insert_writes_few_nodes() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        // Build a 1000-entry tree.
        let entries: Vec<Entry> = (0..1000u32)
            .map(|i| make_entry(
                format!("k-{i:06}").as_bytes(),
                format!("v-{i}").as_bytes(),
                dir.path(),
            ))
            .collect();
        let root = build_tree(&entries, &config, dir.path()).unwrap();

        // Count CAS files before.
        let commits_dir = dir.path().join("commits");
        let before_count = std::fs::read_dir(&commits_dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .map_or(false, |ext| ext == "bin")
            })
            .count();

        // Insert one entry.
        let new_entry = make_entry(b"k-000500-new", b"inserted", dir.path());
        let leaf_entry = LeafEntry::from_entry(&new_entry);
        incremental_insert(dir.path(), root, &leaf_entry, &config).unwrap();

        // Count CAS files after.
        let after_count = std::fs::read_dir(&commits_dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .map_or(false, |ext| ext == "bin")
            })
            .count();

        let nodes_written = after_count - before_count;

        // For a 1000-entry tree (~3-4 levels), should write ~3-8 nodes,
        // not 50+ (which a full rebuild would produce).
        assert!(
            nodes_written <= 15,
            "incremental insert should write few nodes, got {nodes_written}"
        );
        assert!(
            nodes_written >= 2,
            "should write at least a leaf and a branch, got {nodes_written}"
        );
    }
}

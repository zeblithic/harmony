// Incremental O(log n) Prolly Tree mutations.
//
// Instead of rebuilding the entire tree from scratch on every mutation,
// walk from root to the affected leaf, mutate, rechunk the affected
// region until boundaries re-converge, and cascade new CIDs up to root.

use crate::error::DbError;
use harmony_content::ContentId;
use std::path::Path;
use super::chunker::ChunkerConfig;
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
}

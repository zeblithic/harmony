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
}

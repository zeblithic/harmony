// Recursive O(d) tree diff.
//
// Compares two Prolly Tree roots, skipping identical subtrees in O(1) when
// the child CID matches. Only the "spine" of changed nodes is traversed,
// giving O(d) I/O where d is the number of differing leaves.

use crate::error::DbError;
use crate::types::{Entry, TableDiff};
use harmony_content::ContentId;
use super::node::{Node, LeafEntry, BranchEntry};
use std::cmp::Ordering;
use std::path::Path;

/// Public entry point: diff two Prolly Tree roots and return a `TableDiff`.
pub(crate) fn diff_trees(
    data_dir: &Path,
    root_a: Option<ContentId>,
    root_b: Option<ContentId>,
) -> Result<TableDiff, DbError> {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();

    match (root_a, root_b) {
        (None, None) => {}
        (Some(cid_a), None) => {
            collect_all(data_dir, cid_a, &mut removed)?;
        }
        (None, Some(cid_b)) => {
            collect_all(data_dir, cid_b, &mut added)?;
        }
        (Some(cid_a), Some(cid_b)) => {
            if cid_a == cid_b {
                // Identical roots — O(1) skip.
            } else {
                diff_nodes(data_dir, cid_a, cid_b, &mut added, &mut removed, &mut changed)?;
            }
        }
    }

    Ok(TableDiff { added, removed, changed })
}

/// Recursive dispatch: load both nodes and diff based on their types.
fn diff_nodes(
    data_dir: &Path,
    cid_a: ContentId,
    cid_b: ContentId,
    added: &mut Vec<Entry>,
    removed: &mut Vec<Entry>,
    changed: &mut Vec<(Entry, Entry)>,
) -> Result<(), DbError> {
    let node_a = Node::read_from_cas(data_dir, cid_a)?;
    let node_b = Node::read_from_cas(data_dir, cid_b)?;

    match (node_a, node_b) {
        (Node::Leaf(entries_a), Node::Leaf(entries_b)) => {
            diff_leaf_entries(&entries_a, &entries_b, added, removed, changed);
        }
        (Node::Branch(children_a), Node::Branch(children_b)) => {
            diff_branches(data_dir, &children_a, &children_b, added, removed, changed)?;
        }
        // Depth mismatch: flatten both subtrees to sorted entries and merge.
        (node_a, node_b) => {
            let mut flat_a = Vec::new();
            collect_raw_leaves(data_dir, &node_a, &mut flat_a)?;
            let mut flat_b = Vec::new();
            collect_raw_leaves(data_dir, &node_b, &mut flat_b)?;
            diff_leaf_entries(&flat_a, &flat_b, added, removed, changed);
        }
    }

    Ok(())
}

/// Two-pointer walk on branch entries (sorted by boundary_key).
fn diff_branches(
    data_dir: &Path,
    children_a: &[BranchEntry],
    children_b: &[BranchEntry],
    added: &mut Vec<Entry>,
    removed: &mut Vec<Entry>,
    changed: &mut Vec<(Entry, Entry)>,
) -> Result<(), DbError> {
    let mut i = 0;
    let mut j = 0;

    while i < children_a.len() && j < children_b.len() {
        let a = &children_a[i];
        let b = &children_b[j];

        match a.boundary_key.cmp(&b.boundary_key) {
            Ordering::Equal => {
                if a.child_cid != b.child_cid {
                    diff_nodes(
                        data_dir,
                        ContentId::from_bytes(a.child_cid),
                        ContentId::from_bytes(b.child_cid),
                        added, removed, changed,
                    )?;
                }
                i += 1;
                j += 1;
            }
            _ => {
                // Boundaries don't align — chunk boundaries shifted.
                // Collect leaf entries from mismatched region on both sides,
                // then diff them as flat entry lists.
                let mut entries_a = Vec::new();
                let mut entries_b = Vec::new();

                // Advance both sides until boundaries re-align or one side ends.
                // We need to consume subtrees from both sides that overlap in
                // key range, then diff the collected entries.
                //
                // Strategy: find the sync point — the next boundary key that
                // appears in both sides. Collect everything before it.
                let sync_key = find_sync_key(&children_a[i..], &children_b[j..]);

                while i < children_a.len() {
                    if let Some(ref sk) = sync_key {
                        if children_a[i].boundary_key > *sk { break; }
                    }
                    collect_raw_leaves_by_cid(data_dir, ContentId::from_bytes(children_a[i].child_cid), &mut entries_a)?;
                    i += 1;
                    if let Some(ref sk) = sync_key {
                        if i < children_a.len() && children_a[i - 1].boundary_key == *sk { break; }
                    }
                }

                while j < children_b.len() {
                    if let Some(ref sk) = sync_key {
                        if children_b[j].boundary_key > *sk { break; }
                    }
                    collect_raw_leaves_by_cid(data_dir, ContentId::from_bytes(children_b[j].child_cid), &mut entries_b)?;
                    j += 1;
                    if let Some(ref sk) = sync_key {
                        if j < children_b.len() && children_b[j - 1].boundary_key == *sk { break; }
                    }
                }

                diff_leaf_entries(&entries_a, &entries_b, added, removed, changed);
            }
        }
    }

    // Remaining in A are all removed.
    while i < children_a.len() {
        collect_all(data_dir, ContentId::from_bytes(children_a[i].child_cid), removed)?;
        i += 1;
    }
    // Remaining in B are all added.
    while j < children_b.len() {
        collect_all(data_dir, ContentId::from_bytes(children_b[j].child_cid), added)?;
        j += 1;
    }

    Ok(())
}

/// Find the next boundary key that appears in both remaining slices.
fn find_sync_key(a: &[BranchEntry], b: &[BranchEntry]) -> Option<Vec<u8>> {
    // Skip the first entries (they're the ones that don't match).
    for ea in a.iter().skip(1) {
        for eb in b.iter().skip(1) {
            if ea.boundary_key == eb.boundary_key {
                return Some(ea.boundary_key.clone());
            }
        }
    }
    None
}

/// Two-pointer merge on sorted leaf entries.
///
/// An entry is "changed" when the key matches but `value_cid`, `flags`, or
/// `snippet` differs. Timestamp is intentionally excluded from comparison.
fn diff_leaf_entries(
    entries_a: &[LeafEntry],
    entries_b: &[LeafEntry],
    added: &mut Vec<Entry>,
    removed: &mut Vec<Entry>,
    changed: &mut Vec<(Entry, Entry)>,
) {
    let mut i = 0;
    let mut j = 0;

    while i < entries_a.len() && j < entries_b.len() {
        let a = &entries_a[i];
        let b = &entries_b[j];

        match a.key.cmp(&b.key) {
            Ordering::Equal => {
                // Same key — check if content differs (excluding timestamp).
                if a.value_cid != b.value_cid || a.flags != b.flags || a.snippet != b.snippet {
                    changed.push((a.to_entry(), b.to_entry()));
                }
                i += 1;
                j += 1;
            }
            Ordering::Less => {
                removed.push(a.to_entry());
                i += 1;
            }
            Ordering::Greater => {
                added.push(b.to_entry());
                j += 1;
            }
        }
    }

    // Remaining in A are removed.
    while i < entries_a.len() {
        removed.push(entries_a[i].to_entry());
        i += 1;
    }

    // Remaining in B are added.
    while j < entries_b.len() {
        added.push(entries_b[j].to_entry());
        j += 1;
    }
}

/// Traverse an entire subtree rooted at `cid`, collecting all leaf entries
/// as public `Entry` values.
fn collect_all(
    data_dir: &Path,
    cid: ContentId,
    out: &mut Vec<Entry>,
) -> Result<(), DbError> {
    let node = Node::read_from_cas(data_dir, cid)?;
    collect_leaf_entries(data_dir, &node, out)
}

/// Collect all raw `LeafEntry` values from a subtree by CID (used for
/// boundary-shift flattening in diff_branches).
fn collect_raw_leaves_by_cid(
    data_dir: &Path,
    cid: ContentId,
    out: &mut Vec<LeafEntry>,
) -> Result<(), DbError> {
    let node = Node::read_from_cas(data_dir, cid)?;
    match node {
        Node::Leaf(entries) => out.extend(entries),
        Node::Branch(children) => {
            for child in children {
                collect_raw_leaves_by_cid(data_dir, ContentId::from_bytes(child.child_cid), out)?;
            }
        }
    }
    Ok(())
}

/// Recursively collect all raw `LeafEntry` values from a node (used for
/// depth-mismatch flattening).
fn collect_raw_leaves(
    data_dir: &Path,
    node: &Node,
    out: &mut Vec<LeafEntry>,
) -> Result<(), DbError> {
    match node {
        Node::Leaf(entries) => {
            out.extend_from_slice(entries);
        }
        Node::Branch(children) => {
            for child in children {
                let child_node = Node::read_from_cas(data_dir, ContentId::from_bytes(child.child_cid))?;
                collect_raw_leaves(data_dir, &child_node, out)?;
            }
        }
    }
    Ok(())
}

/// Helper: recursively collect all leaf entries from a node.
fn collect_leaf_entries(
    data_dir: &Path,
    node: &Node,
    out: &mut Vec<Entry>,
) -> Result<(), DbError> {
    match node {
        Node::Leaf(entries) => {
            for e in entries {
                out.push(e.to_entry());
            }
        }
        Node::Branch(children) => {
            for child in children {
                collect_all(data_dir, ContentId::from_bytes(child.child_cid), out)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist;
    use crate::prolly::{build_tree, chunker::ChunkerConfig};
    use crate::types::EntryMeta;

    fn make_entry(key: &[u8], value: &[u8], data_dir: &Path) -> Entry {
        let cid = persist::write_blob(data_dir, value).unwrap();
        Entry {
            key: key.to_vec(),
            value_cid: cid,
            timestamp: 1000,
            metadata: EntryMeta {
                flags: 0,
                snippet: "".to_string(),
            },
        }
    }

    fn setup_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        dir
    }

    #[test]
    fn diff_identical_roots() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();
        let entries = vec![
            make_entry(b"a", b"va", dir.path()),
            make_entry(b"b", b"vb", dir.path()),
            make_entry(b"c", b"vc", dir.path()),
        ];
        let root = build_tree(&entries, &config, dir.path()).unwrap();

        let diff = diff_trees(dir.path(), root, root).unwrap();
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn diff_additions() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        let entries1 = vec![make_entry(b"a", b"va", dir.path())];
        let root1 = build_tree(&entries1, &config, dir.path()).unwrap();

        let entries2 = vec![
            make_entry(b"a", b"va", dir.path()),
            make_entry(b"b", b"vb", dir.path()),
        ];
        let root2 = build_tree(&entries2, &config, dir.path()).unwrap();

        let diff = diff_trees(dir.path(), root1, root2).unwrap();
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added[0].key, b"b");
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn diff_removals() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        let entries1 = vec![
            make_entry(b"a", b"va", dir.path()),
            make_entry(b"b", b"vb", dir.path()),
        ];
        let root1 = build_tree(&entries1, &config, dir.path()).unwrap();

        let entries2 = vec![make_entry(b"a", b"va", dir.path())];
        let root2 = build_tree(&entries2, &config, dir.path()).unwrap();

        let diff = diff_trees(dir.path(), root1, root2).unwrap();
        assert!(diff.added.is_empty());
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed[0].key, b"b");
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn diff_changes() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        let entries1 = vec![make_entry(b"k", b"old", dir.path())];
        let root1 = build_tree(&entries1, &config, dir.path()).unwrap();

        let entries2 = vec![make_entry(b"k", b"new", dir.path())];
        let root2 = build_tree(&entries2, &config, dir.path()).unwrap();

        let diff = diff_trees(dir.path(), root1, root2).unwrap();
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert_eq!(diff.changed.len(), 1);
        let (old, new) = &diff.changed[0];
        assert_eq!(old.key, b"k");
        assert_eq!(new.key, b"k");
        assert_ne!(old.value_cid, new.value_cid);
    }

    #[test]
    fn diff_none_to_some() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        let entries = vec![
            make_entry(b"x", b"vx", dir.path()),
            make_entry(b"y", b"vy", dir.path()),
        ];
        let root = build_tree(&entries, &config, dir.path()).unwrap();

        let diff = diff_trees(dir.path(), None, root).unwrap();
        assert_eq!(diff.added.len(), 2);
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());

        // Reverse direction: Some → None should be all removed.
        let diff2 = diff_trees(dir.path(), root, None).unwrap();
        assert!(diff2.added.is_empty());
        assert_eq!(diff2.removed.len(), 2);
        assert!(diff2.changed.is_empty());
    }

    #[test]
    fn diff_boundary_shift_no_false_changes() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let config = ChunkerConfig::default_4k();

        let entries_a: Vec<Entry> = (0..200u32)
            .map(|i| make_entry(format!("k-{i:06}").as_bytes(), format!("v-{i}").as_bytes(), dir.path()))
            .collect();
        let root_a = build_tree(&entries_a, &config, dir.path()).unwrap();

        // Same 200 entries + 1 new one in the middle.
        let mut entries_b = entries_a.clone();
        entries_b.push(make_entry(b"k-000100-new", b"inserted", dir.path()));
        entries_b.sort_by(|a, b| a.key.cmp(&b.key));
        let root_b = build_tree(&entries_b, &config, dir.path()).unwrap();

        let td = diff_trees(dir.path(), root_a, root_b).unwrap();
        assert_eq!(td.added.len(), 1, "should detect exactly 1 addition, not spurious changes from boundary shifts");
        assert_eq!(td.added[0].key, b"k-000100-new");
        assert!(td.removed.is_empty(), "no entries were removed");
        assert!(td.changed.is_empty(), "no entries were changed");
    }

    #[test]
    fn diff_large_dataset_small_change() {
        let dir = setup_dir();
        let config = ChunkerConfig::default_4k();

        let entries1: Vec<Entry> = (0..500)
            .map(|i| {
                let key = format!("entry-{i:04}").into_bytes();
                let val = format!("value-{i:04}").into_bytes();
                make_entry(&key, &val, dir.path())
            })
            .collect();
        let root1 = build_tree(&entries1, &config, dir.path()).unwrap();

        // Change only entry 250.
        let mut entries2 = entries1.clone();
        entries2[250] = make_entry(
            &format!("entry-{:04}", 250).into_bytes(),
            b"CHANGED",
            dir.path(),
        );
        let root2 = build_tree(&entries2, &config, dir.path()).unwrap();

        assert_ne!(root1, root2, "roots should differ after a value change");

        let diff = diff_trees(dir.path(), root1, root2).unwrap();
        assert!(
            diff.added.is_empty(),
            "no entries were added, got {} additions",
            diff.added.len()
        );
        assert!(
            diff.removed.is_empty(),
            "no entries were removed, got {} removals",
            diff.removed.len()
        );
        assert_eq!(
            diff.changed.len(),
            1,
            "exactly one entry changed, got {} changes",
            diff.changed.len()
        );
        let (old, new) = &diff.changed[0];
        assert_eq!(old.key, format!("entry-{:04}", 250).into_bytes());
        assert_eq!(new.key, format!("entry-{:04}", 250).into_bytes());
        assert_ne!(old.value_cid, new.value_cid);
    }
}

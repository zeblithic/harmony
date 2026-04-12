use crate::error::DbError;
use crate::persist;
use crate::table::Table;
use crate::types::Entry;
use harmony_content::book::BookStore;
use harmony_content::{ContentFlags, ContentId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

const COMMIT_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RootManifest {
    version: u32,
    #[serde(
        serialize_with = "crate::types::ser_opt_cid",
        deserialize_with = "crate::types::de_opt_cid"
    )]
    parent: Option<ContentId>,
    tables: BTreeMap<String, String>,
}

pub(crate) fn create_commit(
    data_dir: &Path,
    head: Option<ContentId>,
    tables: &HashMap<String, Table>,
    mut store: Option<&mut dyn BookStore>,
) -> Result<ContentId, DbError> {
    let commits_dir = data_dir.join("commits");
    let mut table_cids: BTreeMap<String, String> = BTreeMap::new();

    for (name, table) in tables {
        let page_bytes = serde_json::to_vec(table.entries())
            .map_err(|e| DbError::Serialize(e.to_string()))?;
        let page_cid = ContentId::for_book(&page_bytes, ContentFlags::default())
            .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
        let page_hex = hex::encode(page_cid.to_bytes());

        let page_path = commits_dir.join(format!("{page_hex}.bin"));
        if !page_path.exists() {
            let tmp = commits_dir.join(format!("{page_hex}.bin.tmp"));
            std::fs::write(&tmp, &page_bytes)?;
            std::fs::rename(&tmp, &page_path)?;
        }

        if let Some(ref mut s) = store {
            s.insert_with_flags(&page_bytes, ContentFlags::default())
                .map_err(|e| DbError::Serialize(format!("BookStore page push failed: {e:?}")))?;
        }

        table_cids.insert(name.clone(), page_hex);
    }

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

    let root_path = commits_dir.join(format!("{root_hex}.bin"));
    if !root_path.exists() {
        let tmp = commits_dir.join(format!("{root_hex}.bin.tmp"));
        std::fs::write(&tmp, &manifest_bytes)?;
        std::fs::rename(&tmp, &root_path)?;
    }

    if let Some(ref mut s) = store {
        // Push value blobs before manifest so the manifest never references
        // missing data in the remote store.
        for table in tables.values() {
            for entry in table.entries() {
                if !s.contains(&entry.value_cid) {
                    match persist::read_blob(data_dir, &entry.value_cid)? {
                        Some(blob) => s.store(entry.value_cid, blob),
                        None => {} // Blob not on disk (orphaned) — best-effort skip
                    }
                }
            }
        }
        // Manifest last — makes the root CID discoverable only after all
        // referenced data is available.
        s.insert_with_flags(&manifest_bytes, ContentFlags::default())
            .map_err(|e| DbError::Serialize(format!("BookStore manifest push failed: {e:?}")))?;
    }

    Ok(root_cid)
}

pub(crate) fn load_manifest(
    data_dir: &Path,
    root_cid: ContentId,
    store: Option<&dyn BookStore>,
) -> Result<RootManifest, DbError> {
    let root_hex = hex::encode(root_cid.to_bytes());
    let local_path = data_dir.join("commits").join(format!("{root_hex}.bin"));

    let (bytes, from_store) = if local_path.exists() {
        (std::fs::read(&local_path)?, false)
    } else if let Some(s) = store {
        let fetched = s.get(&root_cid)
            .map(|b| b.to_vec())
            .ok_or_else(|| DbError::CommitNotFound { cid: root_hex.clone() })?;
        (fetched, true)
    } else {
        return Err(DbError::CommitNotFound { cid: root_hex });
    };

    // Verify content hashes to expected CID (rejects corrupt/adversarial stores).
    if from_store {
        let computed = ContentId::for_book(&bytes, ContentFlags::default())
            .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
        if computed != root_cid {
            return Err(DbError::CorruptIndex(format!(
                "manifest content mismatch for {root_hex}"
            )));
        }
    }

    let manifest: RootManifest = serde_json::from_slice(&bytes)
        .map_err(|e| DbError::CorruptIndex(e.to_string()))?;
    if manifest.version != COMMIT_VERSION {
        return Err(DbError::CorruptIndex(format!(
            "unsupported commit version: {} (expected {})",
            manifest.version, COMMIT_VERSION
        )));
    }

    // Cache locally after validation so diff() works without a store later.
    if from_store {
        let tmp = local_path.with_extension("bin.tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &local_path)?;
    }

    Ok(manifest)
}

fn load_page(
    data_dir: &Path,
    page_hex: &str,
    store: Option<&dyn BookStore>,
) -> Result<Vec<Entry>, DbError> {
    // Validate hex format upfront (prevents path traversal from corrupt manifests).
    let page_bytes: [u8; 32] = hex::decode(page_hex)
        .map_err(|e| DbError::CorruptIndex(e.to_string()))?
        .try_into()
        .map_err(|_| DbError::CorruptIndex("bad page CID length".into()))?;
    let canonical_hex = hex::encode(page_bytes);
    let local_path = data_dir.join("commits").join(format!("{canonical_hex}.bin"));

    let (bytes, from_store) = if local_path.exists() {
        (std::fs::read(&local_path)?, false)
    } else if let Some(s) = store {
        let page_cid = ContentId::from_bytes(page_bytes);
        let fetched = s.get(&page_cid)
            .map(|b| b.to_vec())
            .ok_or(DbError::CommitNotFound { cid: canonical_hex })?;
        (fetched, true)
    } else {
        return Err(DbError::CommitNotFound { cid: canonical_hex });
    };

    // Verify content hashes to expected CID (rejects corrupt/adversarial stores).
    if from_store {
        let computed = ContentId::for_book(&bytes, ContentFlags::default())
            .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
        let expected_cid = ContentId::from_bytes(page_bytes);
        if computed != expected_cid {
            return Err(DbError::CorruptIndex(format!(
                "page content mismatch for {}", hex::encode(page_bytes)
            )));
        }
    }

    let entries: Vec<Entry> = serde_json::from_slice(&bytes)
        .map_err(|e| DbError::CorruptIndex(e.to_string()))?;

    if from_store {
        let tmp = local_path.with_extension("bin.tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &local_path)?;
    }

    Ok(entries)
}

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
            let cid_hex = hex::encode(entry.value_cid.to_bytes());
            let local_blob = data_dir.join("blobs").join(format!("{cid_hex}.bin"));
            if !local_blob.exists() {
                if let Some(s) = store {
                    if let Some(blob_data) = s.get(&entry.value_cid) {
                        persist::write_blob_raw(data_dir, &cid_hex, blob_data)?;
                    }
                }
            }
            table.upsert(entry);
        }
        tables.insert(name.clone(), table);
    }

    Ok(tables)
}

pub(crate) fn diff_commits(
    data_dir: &Path,
    old_cid: ContentId,
    new_cid: ContentId,
    store: Option<&dyn BookStore>,
) -> Result<crate::types::Diff, DbError> {
    let old_manifest = load_manifest(data_dir, old_cid, store)?;
    let new_manifest = load_manifest(data_dir, new_cid, store)?;

    let mut tables = HashMap::new();

    for (name, new_page_hex) in &new_manifest.tables {
        if let Some(old_page_hex) = old_manifest.tables.get(name) {
            if old_page_hex == new_page_hex {
                continue;
            }
            let old_entries = load_page(data_dir, old_page_hex, store)?;
            let new_entries = load_page(data_dir, new_page_hex, store)?;
            let td = diff_entries(&old_entries, &new_entries);
            if !td.added.is_empty() || !td.removed.is_empty() || !td.changed.is_empty() {
                tables.insert(name.clone(), td);
            }
        } else {
            let new_entries = load_page(data_dir, new_page_hex, store)?;
            tables.insert(name.clone(), crate::types::TableDiff {
                added: new_entries,
                removed: Vec::new(),
                changed: Vec::new(),
            });
        }
    }

    for (name, old_page_hex) in &old_manifest.tables {
        if !new_manifest.tables.contains_key(name) {
            let old_entries = load_page(data_dir, old_page_hex, store)?;
            tables.insert(name.clone(), crate::types::TableDiff {
                added: Vec::new(),
                removed: old_entries,
                changed: Vec::new(),
            });
        }
    }

    Ok(crate::types::Diff { tables })
}

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
                if old[oi].value_cid != new[ni].value_cid
                    || old[oi].metadata != new[ni].metadata {
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
        let root2 = create_commit(dir.path(), None, &tables, None).unwrap();
        assert_eq!(root, root2);
    }

    #[test]
    fn commit_with_bookstore() {
        use harmony_content::book::MemoryBookStore;
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
        use harmony_content::book::MemoryBookStore;
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

        // Delete local commits to force BookStore fallback.
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

        let rebuilt = rebuild(dir.path(), root, None).unwrap();
        assert_eq!(rebuilt["t"].len(), 1); // Entry visible, blob missing.
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
        let d = diff_commits(dir.path(), root, root, None).unwrap();
        assert!(d.tables.is_empty());
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

        let d = diff_commits(dir.path(), root1, root2, None).unwrap();
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

        let d = diff_commits(dir.path(), root1, root2, None).unwrap();
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

        let d = diff_commits(dir.path(), root1, root2, None).unwrap();
        assert_eq!(d.tables["t"].changed.len(), 1);
        assert_eq!(d.tables["t"].changed[0].0.value_cid, cid_old);
        assert_eq!(d.tables["t"].changed[0].1.value_cid, cid_new);
    }

    #[test]
    fn diff_metadata_changes() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let cid = ContentId::for_book(b"val", ContentFlags::default()).unwrap();

        let mut tables1 = HashMap::new();
        let mut t1 = Table::new();
        t1.upsert(Entry {
            key: b"k".to_vec(), value_cid: cid, timestamp: 1,
            metadata: EntryMeta { flags: 0, snippet: "unread".to_string() },
        });
        tables1.insert("t".to_string(), t1);
        let root1 = create_commit(dir.path(), None, &tables1, None).unwrap();

        let mut tables2 = HashMap::new();
        let mut t2 = Table::new();
        t2.upsert(Entry {
            key: b"k".to_vec(), value_cid: cid, timestamp: 1,
            metadata: EntryMeta { flags: 1, snippet: "read".to_string() },
        });
        tables2.insert("t".to_string(), t2);
        let root2 = create_commit(dir.path(), Some(root1), &tables2, None).unwrap();

        let d = diff_commits(dir.path(), root1, root2, None).unwrap();
        assert_eq!(d.tables["t"].changed.len(), 1);
        assert_eq!(d.tables["t"].changed[0].0.metadata.flags, 0);
        assert_eq!(d.tables["t"].changed[0].1.metadata.flags, 1);
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

        let cid2 = ContentId::for_book(b"v2", ContentFlags::default()).unwrap();
        tables.get_mut("changed").unwrap().upsert(Entry {
            key: b"y".to_vec(), value_cid: cid2, timestamp: 2, metadata: meta(""),
        });
        let root2 = create_commit(dir.path(), Some(root1), &tables, None).unwrap();

        let d = diff_commits(dir.path(), root1, root2, None).unwrap();
        assert!(!d.tables.contains_key("unchanged"));
        assert!(d.tables.contains_key("changed"));
    }

    #[test]
    fn diff_after_rebuild_from_bookstore() {
        // Regression test for P1 bug: rebuild fetches commit blobs from
        // BookStore but didn't cache them locally, so diff() failed with
        // CommitNotFound after open_from_cas.
        use harmony_content::book::MemoryBookStore;

        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let cid_a = ContentId::for_book(b"a", ContentFlags::default()).unwrap();
        let cid_b = ContentId::for_book(b"b", ContentFlags::default()).unwrap();

        // Commit 1: one entry.
        let mut tables1 = HashMap::new();
        let mut t1 = Table::new();
        t1.upsert(Entry { key: b"k1".to_vec(), value_cid: cid_a, timestamp: 1, metadata: meta("") });
        tables1.insert("t".to_string(), t1);

        let mut bs = MemoryBookStore::new();
        let root1 = create_commit(dir.path(), None, &tables1, Some(&mut bs)).unwrap();

        // Commit 2: add an entry.
        let mut tables2 = HashMap::new();
        let mut t2 = Table::new();
        t2.upsert(Entry { key: b"k1".to_vec(), value_cid: cid_a, timestamp: 1, metadata: meta("") });
        t2.upsert(Entry { key: b"k2".to_vec(), value_cid: cid_b, timestamp: 2, metadata: meta("") });
        tables2.insert("t".to_string(), t2);
        let root2 = create_commit(dir.path(), Some(root1), &tables2, Some(&mut bs)).unwrap();

        // Rebuild into a fresh directory from BookStore only.
        let dir2 = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir2.path()).unwrap();
        let rebuilt = rebuild(dir2.path(), root2, Some(&bs)).unwrap();
        assert_eq!(rebuilt["t"].len(), 2);

        // diff() works: root2 was cached locally by rebuild, root1 fetched
        // from BookStore via the store parameter.
        let d = diff_commits(dir2.path(), root1, root2, Some(&bs)).unwrap();
        assert_eq!(d.tables["t"].added.len(), 1);
        assert_eq!(d.tables["t"].added[0].key, b"k2");
    }

    #[test]
    fn diff_new_table_from_bookstore() {
        // Exercises the diff code path where a brand-new table exists
        // only in the BookStore (not locally).
        use harmony_content::book::MemoryBookStore;

        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();

        let cid_a = ContentId::for_book(b"a", ContentFlags::default()).unwrap();
        let cid_b = ContentId::for_book(b"b", ContentFlags::default()).unwrap();

        // Commit 1: only table "t".
        let mut tables1 = HashMap::new();
        let mut t1 = Table::new();
        t1.upsert(Entry { key: b"k1".to_vec(), value_cid: cid_a, timestamp: 1, metadata: meta("") });
        tables1.insert("t".to_string(), t1);

        let mut bs = MemoryBookStore::new();
        let root1 = create_commit(dir.path(), None, &tables1, Some(&mut bs)).unwrap();

        // Commit 2: add new table "new_t".
        let mut tables2 = tables1.clone();
        let mut t2 = Table::new();
        t2.upsert(Entry { key: b"n1".to_vec(), value_cid: cid_b, timestamp: 2, metadata: meta("") });
        tables2.insert("new_t".to_string(), t2);
        let root2 = create_commit(dir.path(), Some(root1), &tables2, Some(&mut bs)).unwrap();

        // Rebuild in fresh directory, then diff.
        let dir2 = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir2.path()).unwrap();
        let _rebuilt = rebuild(dir2.path(), root2, Some(&bs)).unwrap();

        let d = diff_commits(dir2.path(), root1, root2, Some(&bs)).unwrap();
        // "t" unchanged → not in diff. "new_t" is brand new → all entries added.
        assert!(!d.tables.contains_key("t"));
        assert_eq!(d.tables["new_t"].added.len(), 1);
        assert_eq!(d.tables["new_t"].added[0].key, b"n1");
    }
}

use crate::error::DbError;
use harmony_content::{ContentFlags, ContentId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

const ROOTS_VERSION: u32 = 2;
const MAX_SNIPPET_BYTES: usize = 256;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RootsFile {
    pub version: u32,
    #[serde(
        serialize_with = "crate::types::ser_opt_cid",
        deserialize_with = "crate::types::de_opt_cid"
    )]
    pub head: Option<ContentId>,
    pub table_roots: BTreeMap<String, String>, // name → root CID hex
}

pub(crate) fn load_roots(
    data_dir: &Path,
) -> (Option<ContentId>, BTreeMap<String, ContentId>) {
    let path = data_dir.join("index.json");
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return (None, BTreeMap::new()),
    };
    let rf: RootsFile = match serde_json::from_slice::<RootsFile>(&bytes) {
        Ok(r) if r.version == ROOTS_VERSION => r,
        _ => return (None, BTreeMap::new()),
    };
    let table_roots = rf
        .table_roots
        .into_iter()
        .filter_map(|(name, hex_str)| {
            let bytes: [u8; 32] = hex::decode(&hex_str).ok()?.try_into().ok()?;
            Some((name, ContentId::from_bytes(bytes)))
        })
        .collect();
    (rf.head, table_roots)
}

pub(crate) fn save_roots(
    data_dir: &Path,
    head: Option<ContentId>,
    table_roots: &BTreeMap<String, Option<ContentId>>,
) -> Result<(), DbError> {
    let filtered: BTreeMap<String, String> = table_roots
        .iter()
        .filter_map(|(name, opt_cid)| {
            opt_cid.map(|cid| (name.clone(), hex::encode(cid.to_bytes())))
        })
        .collect();
    let rf = RootsFile {
        version: ROOTS_VERSION,
        head,
        table_roots: filtered,
    };
    let bytes =
        serde_json::to_vec_pretty(&rf).map_err(|e| DbError::Serialize(e.to_string()))?;
    atomic_write(&data_dir.join("index.json"), &bytes)
}

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

pub(crate) fn read_blob(data_dir: &Path, cid: &ContentId) -> Result<Option<Vec<u8>>, DbError> {
    let cid_hex = hex::encode(cid.to_bytes());
    let blob_path = data_dir.join("blobs").join(format!("{cid_hex}.bin"));
    match std::fs::read(&blob_path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(DbError::Io(e)),
    }
}

/// Write raw bytes to a blob file by hex CID (used during rebuild).
/// Verifies the content matches the expected CID to reject corrupt/adversarial data.
pub(crate) fn write_blob_raw(data_dir: &Path, cid_hex: &str, data: &[u8]) -> Result<(), DbError> {
    let blob_path = data_dir.join("blobs").join(format!("{cid_hex}.bin"));
    if blob_path.exists() {
        return Ok(());
    }
    // Verify content matches the expected CID before persisting.
    let computed = ContentId::for_book(data, ContentFlags::default())
        .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
    if hex::encode(computed.to_bytes()) != cid_hex {
        return Err(DbError::CorruptIndex(format!(
            "blob content mismatch for {cid_hex}"
        )));
    }
    let tmp = data_dir.join("blobs").join(format!("{cid_hex}.bin.tmp"));
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, &blob_path)?;
    Ok(())
}

pub(crate) fn ensure_dirs(data_dir: &Path) -> Result<(), DbError> {
    std::fs::create_dir_all(data_dir.join("blobs"))?;
    std::fs::create_dir_all(data_dir.join("commits"))?;
    Ok(())
}

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
    fn save_and_load_roots_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        ensure_dirs(dir.path()).unwrap();
        let cid = ContentId::for_book(b"val", ContentFlags::default()).unwrap();
        let mut table_roots = BTreeMap::new();
        table_roots.insert("inbox".to_string(), Some(cid));
        save_roots(dir.path(), None, &table_roots).unwrap();
        let (head, loaded) = load_roots(dir.path());
        assert!(head.is_none());
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded["inbox"], cid);
    }

    #[test]
    fn load_missing_roots_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let (head, table_roots) = load_roots(dir.path());
        assert!(head.is_none());
        assert!(table_roots.is_empty());
    }

    #[test]
    fn load_corrupt_roots_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("index.json"), b"not json{{{").unwrap();
        let (head, table_roots) = load_roots(dir.path());
        assert!(head.is_none());
        assert!(table_roots.is_empty());
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
        let s = "é".repeat(150); // 300 bytes
        let truncated = truncate_snippet(&s);
        assert!(truncated.len() <= MAX_SNIPPET_BYTES);
        assert!(truncated.is_char_boundary(truncated.len()));
    }
}

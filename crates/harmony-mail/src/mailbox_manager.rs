//! Merkle mailbox manager — maintains per-user CAS mailbox trees.
//!
//! Each registered user gets a MailRoot with 4 folders (inbox, sent, drafts,
//! trash). On SMTP delivery, the inbox tree is updated and the new root CID
//! is published to Zenoh for harmony-client consumption.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::mailbox::{
    FolderKind, MailFolder, MailPage, MailRoot, FOLDER_COUNT, MAX_SNIPPET_LEN, PAGE_CAPACITY,
};
use crate::message::{ADDRESS_HASH_LEN, CID_LEN, MESSAGE_ID_LEN};

/// Errors from mailbox manager operations.
#[derive(Debug, thiserror::Error)]
pub enum MailboxError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("CAS error: {0}")]
    Cas(String),
    #[error("mailbox format error: {0}")]
    Format(#[from] crate::error::MailError),
    #[error("no mailbox for user {}", hex::encode(.0))]
    NoMailbox([u8; ADDRESS_HASH_LEN]),
}

pub struct MailboxManager {
    /// Per-user current root CIDs.
    roots: HashMap<[u8; ADDRESS_HASH_LEN], [u8; CID_LEN]>,
    /// CAS storage path for DiskBookStore access.
    content_store_path: PathBuf,
    /// Persistence for root CID pointers.
    db: rusqlite::Connection,
}

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS mailbox_roots (
    address BLOB NOT NULL UNIQUE,
    root_cid BLOB NOT NULL
);
";

impl MailboxManager {
    /// Open or create the mailbox roots database.
    pub fn open(db_path: &Path, content_store_path: &Path) -> Result<Self, MailboxError> {
        let db = rusqlite::Connection::open(db_path)?;
        db.execute_batch(SCHEMA_SQL)?;

        // Load existing roots into memory.
        // The statement borrow must be dropped before `db` is moved into Self,
        // so we collect all rows eagerly inside a nested block.
        let mut roots = HashMap::new();
        {
            let mut stmt = db.prepare("SELECT address, root_cid FROM mailbox_roots")?;
            let pairs: Vec<(Vec<u8>, Vec<u8>)> = stmt
                .query_map([], |row| {
                    let addr_blob: Vec<u8> = row.get(0)?;
                    let cid_blob: Vec<u8> = row.get(1)?;
                    Ok((addr_blob, cid_blob))
                })?
                .collect::<Result<_, _>>()?;
            for (addr_blob, cid_blob) in pairs {
                if addr_blob.len() == ADDRESS_HASH_LEN && cid_blob.len() == CID_LEN {
                    let mut addr = [0u8; ADDRESS_HASH_LEN];
                    addr.copy_from_slice(&addr_blob);
                    let mut cid = [0u8; CID_LEN];
                    cid.copy_from_slice(&cid_blob);
                    roots.insert(addr, cid);
                }
            }
        }

        Ok(Self {
            roots,
            content_store_path: content_store_path.to_path_buf(),
            db,
        })
    }

    /// Get the current root CID for a user, if one exists.
    pub fn get_root(&self, address: &[u8; ADDRESS_HASH_LEN]) -> Option<&[u8; CID_LEN]> {
        self.roots.get(address)
    }

    /// Persist a root CID for a user (inserts or updates).
    fn persist_root(
        &mut self,
        address: &[u8; ADDRESS_HASH_LEN],
        root_cid: &[u8; CID_LEN],
    ) -> Result<(), MailboxError> {
        self.db.execute(
            "INSERT INTO mailbox_roots (address, root_cid) VALUES (?1, ?2)
             ON CONFLICT(address) DO UPDATE SET root_cid = excluded.root_cid",
            rusqlite::params![address.as_slice(), root_cid.as_slice()],
        )?;
        self.roots.insert(*address, *root_cid);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_creates_schema() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        assert!(mgr.roots.is_empty());
    }

    #[test]
    fn persist_and_reload_root() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let cid = [0xBBu8; CID_LEN];

        // Persist
        {
            let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
            mgr.persist_root(&addr, &cid).unwrap();
            assert_eq!(mgr.get_root(&addr), Some(&cid));
        }

        // Reload from disk
        {
            let mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
            assert_eq!(mgr.get_root(&addr), Some(&cid));
        }
    }

    #[test]
    fn persist_root_upserts() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let cid1 = [0x11u8; CID_LEN];
        let cid2 = [0x22u8; CID_LEN];

        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        mgr.persist_root(&addr, &cid1).unwrap();
        assert_eq!(mgr.get_root(&addr), Some(&cid1));

        mgr.persist_root(&addr, &cid2).unwrap();
        assert_eq!(mgr.get_root(&addr), Some(&cid2));
    }
}

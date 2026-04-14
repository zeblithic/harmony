//! Merkle mailbox manager — maintains per-user CAS mailbox trees.
//!
//! Each registered user gets a MailRoot with 4 folders (inbox, sent, drafts,
//! trash). On SMTP delivery, the inbox tree is updated and the new root CID
//! is published to Zenoh for harmony-client consumption.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use harmony_content::book::BookStore as _;

use crate::mailbox::{MailFolder, MailPage, MailRoot, FOLDER_COUNT};
use crate::message::{ADDRESS_HASH_LEN, CID_LEN};

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

    /// Ingest raw bytes into CAS and return the resulting CID as a 32-byte array.
    fn cas_ingest(&self, data: &[u8]) -> Result<[u8; CID_LEN], MailboxError> {
        let mut book = harmony_db::DiskBookStore::new(&self.content_store_path);
        let cid = harmony_content::dag::ingest(
            data,
            &harmony_content::chunker::ChunkerConfig::DEFAULT,
            &mut book,
        )
        .map_err(|e| MailboxError::Cas(e.to_string()))?;
        Ok(cid.to_bytes())
    }

    /// Load raw bytes from CAS by CID.
    #[allow(dead_code)]
    fn cas_load(&self, cid: &[u8; CID_LEN]) -> Result<Vec<u8>, MailboxError> {
        let book = harmony_db::DiskBookStore::new(&self.content_store_path);
        let content_id = harmony_content::cid::ContentId::from_bytes(*cid);
        book.get(&content_id)
            .map(|b: &[u8]| b.to_vec())
            .ok_or_else(|| MailboxError::Cas(format!("CID not found: {}", hex::encode(cid))))
    }

    /// Ensure a user has a Merkle mailbox tree. If one already exists, this is a no-op.
    /// If not, creates an empty MailRoot with 4 empty folders, ingests everything
    /// into CAS, and persists the root CID.
    pub fn ensure_user_mailbox(
        &mut self,
        address: &[u8; ADDRESS_HASH_LEN],
    ) -> Result<(), MailboxError> {
        if self.roots.contains_key(address) {
            return Ok(());
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Create an empty page (shared by all empty folders)
        let empty_page = MailPage::new_empty();
        let page_bytes = empty_page.to_bytes()?;
        let page_cid = self.cas_ingest(&page_bytes)?;

        // Create 4 empty folders, each pointing to the empty page
        let mut folder_cids = [[0u8; CID_LEN]; FOLDER_COUNT];
        for folder_cid in &mut folder_cids {
            let folder = MailFolder {
                version: crate::mailbox::MAILBOX_VERSION,
                message_count: 0,
                unread_count: 0,
                page_cids: vec![page_cid],
            };
            let folder_bytes = folder.to_bytes()?;
            *folder_cid = self.cas_ingest(&folder_bytes)?;
        }

        // Create the root
        let root = MailRoot {
            version: crate::mailbox::MAILBOX_VERSION,
            owner_address: *address,
            updated_at: now,
            folders: folder_cids,
        };
        let root_bytes = root.to_bytes();
        let root_cid = self.cas_ingest(&root_bytes)?;

        self.persist_root(address, &root_cid)?;
        Ok(())
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
    use crate::mailbox::FolderKind;

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

    #[test]
    fn ensure_user_mailbox_creates_tree() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();

        // No root yet
        assert!(mgr.get_root(&addr).is_none());

        // Create the empty tree
        mgr.ensure_user_mailbox(&addr).unwrap();

        // Root CID now exists
        let root_cid = mgr.get_root(&addr).expect("root should exist");
        assert_ne!(root_cid, &[0u8; CID_LEN], "root CID should not be empty");

        // Verify the root can be loaded from CAS
        let book = harmony_db::DiskBookStore::new(&cas_path);
        let root_content_id = harmony_content::cid::ContentId::from_bytes(*root_cid);
        let root_bytes = book.get(&root_content_id).expect("root should be in CAS");
        let root = MailRoot::from_bytes(root_bytes).unwrap();
        assert_eq!(root.owner_address, addr);

        // Verify inbox folder exists in CAS
        let inbox_cid = root.folder_cid(FolderKind::Inbox);
        assert_ne!(inbox_cid, &[0u8; CID_LEN]);
        let folder_content_id = harmony_content::cid::ContentId::from_bytes(*inbox_cid);
        let folder_bytes = book.get(&folder_content_id).expect("inbox folder should be in CAS");
        let folder = MailFolder::from_bytes(folder_bytes).unwrap();
        assert_eq!(folder.message_count, 0);
        assert_eq!(folder.unread_count, 0);
    }

    #[test]
    fn ensure_user_mailbox_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();

        mgr.ensure_user_mailbox(&addr).unwrap();
        let cid1 = *mgr.get_root(&addr).unwrap();

        // Second call should not change the root CID
        mgr.ensure_user_mailbox(&addr).unwrap();
        let cid2 = *mgr.get_root(&addr).unwrap();
        assert_eq!(cid1, cid2);
    }
}

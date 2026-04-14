//! Merkle mailbox manager — maintains per-user CAS mailbox trees.
//!
//! Each registered user gets a MailRoot with 4 folders (inbox, sent, drafts,
//! trash). On SMTP delivery, the inbox tree is updated and the new root CID
//! is published to Zenoh for harmony-client consumption.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tokio::sync::mpsc;

use crate::mailbox::{FolderKind, MailFolder, MailPage, MailRoot, MessageEntry, FOLDER_COUNT, MAX_SNIPPET_LEN};
use crate::message::{ADDRESS_HASH_LEN, CID_LEN, MESSAGE_ID_LEN};

/// Bounded channel capacity for Zenoh root CID notifications.
///
/// Since only the latest root CID per user is meaningful (older CIDs are
/// superseded), a modest buffer is sufficient. `try_send` drops updates if
/// the queue is full — a dropped notification just delays the client's next
/// refresh, which will pick up the latest root CID on the next successful send.
const ZENOH_NOTIFY_CAPACITY: usize = 128;

/// Notification channel for Zenoh root CID publications.
///
/// `MailboxManager` runs in sync context (`spawn_blocking`). This struct holds
/// the send side of a bounded mpsc channel. A background async task drains
/// the receiver and calls `session.put()`.
///
/// The channel is bounded to provide backpressure: if Zenoh publishing stalls,
/// notifications are dropped (with a throttled warn) rather than accumulating
/// without limit. This is safe because root CID updates are idempotent — the
/// next successful publish will carry the latest root CID regardless of how
/// many intermediate notifications were dropped.
pub struct ZenohPublisher {
    tx: mpsc::Sender<(String, [u8; CID_LEN])>,
}

impl ZenohPublisher {
    /// Create a new publisher backed by a Zenoh session.
    ///
    /// Spawns a background task that drains the channel and publishes
    /// root CID updates to `harmony/messages/{addr_hex}/inbox`.
    pub fn new(session: zenoh::Session) -> Self {
        let (tx, mut rx) = mpsc::channel::<(String, [u8; CID_LEN])>(ZENOH_NOTIFY_CAPACITY);
        tokio::spawn(async move {
            while let Some((addr_hex, root_cid)) = rx.recv().await {
                let topic = format!("harmony/messages/{addr_hex}/inbox");
                if let Err(e) = session.put(&topic, &root_cid[..]).await {
                    tracing::warn!(error = %e, %topic, "Zenoh root CID publish failed");
                }
            }
        });
        Self { tx }
    }

    /// Create a publisher from a raw sender (for testing).
    #[cfg(test)]
    pub fn from_sender(tx: mpsc::Sender<(String, [u8; CID_LEN])>) -> Self {
        Self { tx }
    }

    /// Send a root CID update notification.
    ///
    /// Uses `try_send` to avoid blocking the calling thread. If the channel
    /// is full (Zenoh publishing is stalled), the notification is dropped
    /// and logged. This is safe because root CID updates supersede each other
    /// — the next successful send carries the latest state.
    pub fn notify(&self, addr_hex: String, root_cid: [u8; CID_LEN]) {
        match self.tx.try_send((addr_hex, root_cid)) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!(
                    "ZenohPublisher channel full — root CID notification dropped (client will catch up on next update)"
                );
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::warn!("ZenohPublisher channel closed");
            }
        }
    }
}

/// Errors from mailbox manager operations.
#[derive(Debug, thiserror::Error)]
pub enum ManagerError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("CAS error: {0}")]
    Cas(String),
    #[error("mailbox format error: {0}")]
    Format(#[from] crate::error::MailError),
    #[error("no mailbox for user {}", hex::encode(.0))]
    NoMailbox([u8; ADDRESS_HASH_LEN]),
}

impl From<harmony_mailbox::MailboxError> for ManagerError {
    fn from(e: harmony_mailbox::MailboxError) -> Self {
        ManagerError::Format(crate::error::MailError::Mailbox(e))
    }
}

pub struct MailboxManager {
    /// Per-user current root CIDs.
    roots: HashMap<[u8; ADDRESS_HASH_LEN], [u8; CID_LEN]>,
    /// CAS storage path for DiskBookStore access.
    content_store_path: PathBuf,
    /// Persistence for root CID pointers.
    db: rusqlite::Connection,
    /// Optional Zenoh publisher for root CID notifications.
    publisher: Option<ZenohPublisher>,
}

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS mailbox_roots (
    address BLOB NOT NULL UNIQUE,
    root_cid BLOB NOT NULL
);
";

impl MailboxManager {
    /// Open or create the mailbox roots database.
    pub fn open(db_path: &Path, content_store_path: &Path) -> Result<Self, ManagerError> {
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
                if addr_blob.len() != ADDRESS_HASH_LEN || cid_blob.len() != CID_LEN {
                    return Err(ManagerError::Cas(format!(
                        "corrupt mailbox_roots row: address_len={}, root_cid_len={}",
                        addr_blob.len(),
                        cid_blob.len()
                    )));
                }
                let mut addr = [0u8; ADDRESS_HASH_LEN];
                addr.copy_from_slice(&addr_blob);
                let mut cid = [0u8; CID_LEN];
                cid.copy_from_slice(&cid_blob);
                roots.insert(addr, cid);
            }
        }

        Ok(Self {
            roots,
            content_store_path: content_store_path.to_path_buf(),
            db,
            publisher: None,
        })
    }

    /// Attach a Zenoh publisher for root CID notifications.
    ///
    /// Call this after opening the manager if Zenoh is configured and enabled.
    pub fn set_publisher(&mut self, publisher: ZenohPublisher) {
        self.publisher = Some(publisher);
    }

    /// Get the current root CID for a user, if one exists.
    pub fn get_root(&self, address: &[u8; ADDRESS_HASH_LEN]) -> Option<&[u8; CID_LEN]> {
        self.roots.get(address)
    }

    /// Number of users with initialized Merkle mailboxes.
    pub fn user_count(&self) -> usize {
        self.roots.len()
    }

    /// Ingest raw bytes into CAS and return the resulting CID as a 32-byte array.
    fn cas_ingest(&self, data: &[u8]) -> Result<[u8; CID_LEN], ManagerError> {
        let mut book = harmony_db::DiskBookStore::new(&self.content_store_path);
        let cid = harmony_content::dag::ingest(
            data,
            &harmony_content::chunker::ChunkerConfig::DEFAULT,
            &mut book,
        )
        .map_err(|e| ManagerError::Cas(e.to_string()))?;
        Ok(cid.to_bytes())
    }

    /// Load raw bytes from CAS by CID, reassembling chunked DAG entries.
    fn cas_load(&self, cid: &[u8; CID_LEN]) -> Result<Vec<u8>, ManagerError> {
        let book = harmony_db::DiskBookStore::new(&self.content_store_path);
        let content_id = harmony_content::cid::ContentId::from_bytes(*cid);
        harmony_content::dag::reassemble(&content_id, &book)
            .map_err(|e| ManagerError::Cas(e.to_string()))
    }

    /// Ensure a user has a Merkle mailbox tree. If one already exists, this is a no-op.
    /// If not, creates an empty MailRoot with 4 empty folders, ingests everything
    /// into CAS, and persists the root CID.
    pub fn ensure_user_mailbox(
        &mut self,
        address: &[u8; ADDRESS_HASH_LEN],
    ) -> Result<(), ManagerError> {
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

    /// Insert a message into a user's inbox Merkle tree.
    ///
    /// Loads the current tree from CAS, prepends the entry to the head page
    /// (splitting if full), writes all changed blobs to CAS, and persists the
    /// new root CID.
    pub fn insert_message(
        &mut self,
        user_address: &[u8; ADDRESS_HASH_LEN],
        message_cid: &[u8; CID_LEN],
        msg_id: &[u8; MESSAGE_ID_LEN],
        sender_address: &[u8; ADDRESS_HASH_LEN],
        timestamp: u64,
        subject: &str,
    ) -> Result<(), ManagerError> {
        // Auto-initialize if this user doesn't have a Merkle tree yet
        // (e.g., user created after startup, or roots DB was reset).
        if !self.roots.contains_key(user_address) {
            self.ensure_user_mailbox(user_address)?;
        }
        let root_cid = *self
            .roots
            .get(user_address)
            .ok_or(ManagerError::NoMailbox(*user_address))?;

        // Load current root
        let root_bytes = self.cas_load(&root_cid)?;
        let root = MailRoot::from_bytes(&root_bytes)?;

        // Load inbox folder
        let inbox_cid = *root.folder_cid(FolderKind::Inbox);
        let folder_bytes = self.cas_load(&inbox_cid)?;
        let mut folder = MailFolder::from_bytes(&folder_bytes)?;

        // Build the new entry
        let snippet = crate::mailbox::truncate_utf8(subject, MAX_SNIPPET_LEN).to_string();
        let entry = MessageEntry {
            message_cid: *message_cid,
            message_id: *msg_id,
            sender_address: *sender_address,
            timestamp,
            subject_snippet: snippet,
            read: false,
        };

        // Load head page (or create one if folder has no pages)
        if folder.page_cids.is_empty() {
            // No pages yet — create one with just this entry
            let page = MailPage {
                version: crate::mailbox::MAILBOX_VERSION,
                next_page: None,
                entries: vec![entry],
            };
            let page_cid = self.cas_ingest(&page.to_bytes()?)?;
            folder.page_cids.push(page_cid);
        } else {
            let head_cid = folder.page_cids[0];
            let head_bytes = self.cas_load(&head_cid)?;
            let mut head_page = MailPage::from_bytes(&head_bytes)?;

            if head_page.is_full() {
                // Page is full — create a new head page with just this entry.
                // The old head page becomes the second page (unchanged in CAS).
                let new_page = MailPage {
                    version: crate::mailbox::MAILBOX_VERSION,
                    next_page: Some(head_cid),
                    entries: vec![entry],
                };
                let new_page_cid = self.cas_ingest(&new_page.to_bytes()?)?;
                folder.page_cids.insert(0, new_page_cid);
            } else {
                // Prepend to existing head page
                head_page.entries.insert(0, entry);
                // Update next_page pointer if there's a following page
                head_page.next_page = folder.page_cids.get(1).copied();
                let new_head_cid = self.cas_ingest(&head_page.to_bytes()?)?;
                folder.page_cids[0] = new_head_cid;
            }
        }

        // Update folder counts
        folder.message_count += 1;
        folder.unread_count += 1;

        // Write updated folder to CAS
        let new_folder_cid = self.cas_ingest(&folder.to_bytes()?)?;

        // Update root with new inbox CID
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let new_root = root.with_folder(FolderKind::Inbox, new_folder_cid, now);
        let new_root_cid = self.cas_ingest(&new_root.to_bytes())?;

        // Persist
        self.persist_root(user_address, &new_root_cid)?;

        // Notify Zenoh publisher (non-critical path — errors are logged and swallowed).
        if let Some(ref publisher) = self.publisher {
            let addr_hex = hex::encode(user_address);
            publisher.notify(addr_hex, new_root_cid);
        }

        Ok(())
    }

    /// Persist a root CID for a user (inserts or updates).
    fn persist_root(
        &mut self,
        address: &[u8; ADDRESS_HASH_LEN],
        root_cid: &[u8; CID_LEN],
    ) -> Result<(), ManagerError> {
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
    use harmony_content::book::BookStore as _;
    use crate::mailbox::{FolderKind, PAGE_CAPACITY};
    use crate::message::MESSAGE_ID_LEN;

    fn dummy_msg_cid(tag: u8) -> [u8; CID_LEN] {
        let mut cid = [0u8; CID_LEN];
        cid[0] = tag;
        cid
    }

    fn dummy_msg_id(tag: u8) -> [u8; MESSAGE_ID_LEN] {
        [tag; MESSAGE_ID_LEN]
    }

    fn dummy_sender() -> [u8; ADDRESS_HASH_LEN] {
        [0xEEu8; ADDRESS_HASH_LEN]
    }

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

    #[test]
    fn insert_into_empty_mailbox() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        mgr.ensure_user_mailbox(&addr).unwrap();
        let initial_cid = *mgr.get_root(&addr).unwrap();

        mgr.insert_message(
            &addr,
            &dummy_msg_cid(1),
            &dummy_msg_id(1),
            &dummy_sender(),
            1700000000,
            "Hello World",
        )
        .unwrap();

        // Root CID should have changed
        let new_cid = *mgr.get_root(&addr).unwrap();
        assert_ne!(initial_cid, new_cid);

        // Load and verify the tree
        let book = harmony_db::DiskBookStore::new(&cas_path);
        let root = MailRoot::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(new_cid)).unwrap(),
        )
        .unwrap();

        let inbox_cid = root.folder_cid(FolderKind::Inbox);
        let folder = MailFolder::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*inbox_cid)).unwrap(),
        )
        .unwrap();
        assert_eq!(folder.message_count, 1);
        assert_eq!(folder.unread_count, 1);

        let page = MailPage::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(folder.page_cids[0])).unwrap(),
        )
        .unwrap();
        assert_eq!(page.entries.len(), 1);
        assert_eq!(page.entries[0].message_cid, dummy_msg_cid(1));
        assert_eq!(page.entries[0].subject_snippet, "Hello World");
        assert!(!page.entries[0].read);
    }

    #[test]
    fn insert_splits_page_at_capacity() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        mgr.ensure_user_mailbox(&addr).unwrap();

        // Fill first page to capacity
        for i in 0..PAGE_CAPACITY {
            mgr.insert_message(
                &addr,
                &dummy_msg_cid(i as u8),
                &dummy_msg_id(i as u8),
                &dummy_sender(),
                1700000000 + i as u64,
                &format!("msg {i}"),
            )
            .unwrap();
        }

        // Verify: 1 page with PAGE_CAPACITY entries
        let book = harmony_db::DiskBookStore::new(&cas_path);
        let root_cid = *mgr.get_root(&addr).unwrap();
        let root = MailRoot::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(root_cid)).unwrap(),
        )
        .unwrap();
        let folder = MailFolder::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*root.folder_cid(FolderKind::Inbox))).unwrap(),
        )
        .unwrap();
        assert_eq!(folder.message_count as usize, PAGE_CAPACITY);
        // The initial empty page + PAGE_CAPACITY replacements = still 1 page CID
        assert_eq!(folder.page_cids.len(), 1);

        // Insert one more — should trigger page split
        mgr.insert_message(
            &addr,
            &dummy_msg_cid(0xFF),
            &dummy_msg_id(0xFF),
            &dummy_sender(),
            1700000000 + PAGE_CAPACITY as u64,
            "overflow",
        )
        .unwrap();

        // Verify: 2 pages now
        let root_cid = *mgr.get_root(&addr).unwrap();
        let root = MailRoot::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(root_cid)).unwrap(),
        )
        .unwrap();
        let folder = MailFolder::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*root.folder_cid(FolderKind::Inbox))).unwrap(),
        )
        .unwrap();
        assert_eq!(folder.message_count as usize, PAGE_CAPACITY + 1);
        assert_eq!(folder.page_cids.len(), 2);

        // Head page has 1 entry (the newest)
        let head_page = MailPage::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(folder.page_cids[0])).unwrap(),
        )
        .unwrap();
        assert_eq!(head_page.entries.len(), 1);
        assert_eq!(head_page.entries[0].message_cid, dummy_msg_cid(0xFF));
    }

    #[test]
    fn per_user_isolation() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let alice = [0xAAu8; ADDRESS_HASH_LEN];
        let bob = [0xBBu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        mgr.ensure_user_mailbox(&alice).unwrap();
        mgr.ensure_user_mailbox(&bob).unwrap();

        // Send 3 messages to alice, 1 to bob
        for i in 0..3u8 {
            mgr.insert_message(
                &alice,
                &dummy_msg_cid(i),
                &dummy_msg_id(i),
                &dummy_sender(),
                1700000000 + i as u64,
                &format!("alice msg {i}"),
            )
            .unwrap();
        }
        mgr.insert_message(
            &bob,
            &dummy_msg_cid(10),
            &dummy_msg_id(10),
            &dummy_sender(),
            1700000000,
            "bob msg",
        )
        .unwrap();

        // Verify alice has 3, bob has 1
        let book = harmony_db::DiskBookStore::new(&cas_path);

        let alice_root = MailRoot::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*mgr.get_root(&alice).unwrap())).unwrap(),
        )
        .unwrap();
        let alice_folder = MailFolder::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*alice_root.folder_cid(FolderKind::Inbox))).unwrap(),
        )
        .unwrap();
        assert_eq!(alice_folder.message_count, 3);

        let bob_root = MailRoot::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*mgr.get_root(&bob).unwrap())).unwrap(),
        )
        .unwrap();
        let bob_folder = MailFolder::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*bob_root.folder_cid(FolderKind::Inbox))).unwrap(),
        )
        .unwrap();
        assert_eq!(bob_folder.message_count, 1);
    }

    #[test]
    fn root_cid_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];

        // First session: create tree and insert messages
        {
            let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
            mgr.ensure_user_mailbox(&addr).unwrap();
            for i in 0..3u8 {
                mgr.insert_message(
                    &addr,
                    &dummy_msg_cid(i),
                    &dummy_msg_id(i),
                    &dummy_sender(),
                    1700000000 + i as u64,
                    &format!("msg {i}"),
                )
                .unwrap();
            }
        }

        // Second session: reopen and verify
        {
            let mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
            let root_cid = mgr.get_root(&addr).expect("root should persist");

            let book = harmony_db::DiskBookStore::new(&cas_path);
            let root = MailRoot::from_bytes(
                book.get(&harmony_content::cid::ContentId::from_bytes(*root_cid)).unwrap(),
            )
            .unwrap();
            let folder = MailFolder::from_bytes(
                book.get(&harmony_content::cid::ContentId::from_bytes(*root.folder_cid(FolderKind::Inbox))).unwrap(),
            )
            .unwrap();
            assert_eq!(folder.message_count, 3);

            let page = MailPage::from_bytes(
                book.get(&harmony_content::cid::ContentId::from_bytes(folder.page_cids[0])).unwrap(),
            )
            .unwrap();
            assert_eq!(page.entries.len(), 3);
            // Newest first
            assert_eq!(page.entries[0].message_cid, dummy_msg_cid(2));
            assert_eq!(page.entries[2].message_cid, dummy_msg_cid(0));
        }
    }

    #[tokio::test]
    async fn mailbox_manager_publishes_root_cid() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let (tx, mut rx) = mpsc::channel(16);
        let publisher = ZenohPublisher::from_sender(tx);

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        mgr.set_publisher(publisher);
        mgr.ensure_user_mailbox(&addr).unwrap();

        mgr.insert_message(
            &addr,
            &dummy_msg_cid(1),
            &dummy_msg_id(1),
            &dummy_sender(),
            1700000000,
            "Test Subject",
        )
        .unwrap();

        // Verify the publisher received the notification within a bounded time
        // so a broken notification path fails fast instead of hanging CI.
        let received = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("ZenohPublisher did not notify within 2s")
            .expect("notification channel closed unexpectedly");
        assert_eq!(received.0, hex::encode(addr));
        assert_eq!(received.1, *mgr.get_root(&addr).unwrap());
    }

    #[tokio::test]
    async fn mailbox_manager_no_publish_without_publisher() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xBBu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        // No publisher set — insert_message should succeed silently.
        mgr.insert_message(
            &addr,
            &dummy_msg_cid(2),
            &dummy_msg_id(2),
            &dummy_sender(),
            1700000001,
            "No Publisher",
        )
        .unwrap();
        // Root CID should be present in memory.
        assert!(mgr.get_root(&addr).is_some());
    }
}

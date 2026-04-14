//! Merkle mailbox manager — maintains per-user CAS mailbox trees.
//!
//! Each registered user gets a MailRoot with 4 folders (inbox, sent, drafts,
//! trash). On SMTP delivery, the inbox tree is updated and the new root CID
//! is published to Zenoh for harmony-client consumption.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::mailbox::{FolderKind, MailFolder, MailPage, MailRoot, MessageEntry, FOLDER_COUNT, MAX_SNIPPET_LEN};
use crate::message::{ADDRESS_HASH_LEN, CID_LEN, MESSAGE_ID_LEN};

/// Per-address coalescing publisher for root CID notifications.
///
/// `MailboxManager` runs in sync context (`spawn_blocking`) and calls `notify`
/// whenever a user's mailbox root CID changes. The publisher stores a
/// latest-only map keyed by address hex and wakes a background async drain
/// task. The drain snapshots the map and issues one `session.put()` per user
/// carrying their most recent root CID.
///
/// Why coalesce rather than queue?
/// - No update is lost: `notify` overwrites the entry and wakes the drain, so
///   the drain always observes the newest value per addr.
/// - No log storm: there is no drop path — updates are O(1) mutex-guarded map
///   inserts; storage is bounded by the number of distinct active users, not
///   by update rate.
/// - Stale roots cannot outrun fresh ones: bursts collapse into a single
///   publish of the newest CID rather than replaying a backlog.
pub struct ZenohPublisher {
    latest: Arc<Mutex<HashMap<String, [u8; CID_LEN]>>>,
    wake: Arc<Notify>,
}

impl ZenohPublisher {
    /// Create a new publisher backed by a Zenoh session.
    ///
    /// Spawns a background task that waits for wake-ups, drains the latest
    /// map, and publishes each user's most recent root CID to
    /// `harmony/mail/v1/{addr_hex}/root`.
    ///
    /// Topic namespace: `harmony/mail/v1/*` is the canonical mail prefix
    /// (Phase 0 client consumes `harmony/mail/v1/{addr}` for raw messages
    /// and reserves the `/root` suffix for root CID updates).
    ///
    /// The drain task observes `cancel`: on cancellation the loop exits and
    /// the Zenoh session is dropped explicitly, allowing a clean disconnect
    /// during graceful shutdown rather than holding the socket until the
    /// Tokio runtime is dropped.
    pub fn new(session: zenoh::Session, cancel: CancellationToken) -> Self {
        let latest: Arc<Mutex<HashMap<String, [u8; CID_LEN]>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let wake = Arc::new(Notify::new());

        let drain_latest = Arc::clone(&latest);
        let drain_wake = Arc::clone(&wake);
        tokio::spawn(async move {
            let mut cancelled = false;
            loop {
                if !cancelled {
                    // Wait for a wake-up OR cancellation. If both are pending
                    // simultaneously, `select!` may pick either branch — so
                    // we always drain at least once more AFTER observing cancel
                    // (see the post-loop final-drain pass below) to ensure an
                    // update that raced the cancel signal is still published.
                    tokio::select! {
                        _ = drain_wake.notified() => {}
                        _ = cancel.cancelled() => { cancelled = true; }
                    }
                }
                // Snapshot+clear under the sync lock so notify() can keep
                // inserting while we publish. Held briefly (O(active users));
                // no .await inside this scope.
                let snapshot: Vec<(String, [u8; CID_LEN])> = {
                    let mut map = drain_latest
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    map.drain().collect()
                };
                for (addr_hex, root_cid) in snapshot {
                    let topic = format!("harmony/mail/v1/{addr_hex}/root");
                    if let Err(e) = session.put(&topic, &root_cid[..]).await {
                        tracing::warn!(
                            error = %e,
                            %topic,
                            "Zenoh root CID publish failed"
                        );
                    }
                }
                if cancelled {
                    break;
                }
            }
            // Explicit drop ensures Zenoh disconnects promptly on shutdown
            // rather than waiting for the runtime tear-down.
            drop(session);
            tracing::debug!("ZenohPublisher drain task exited on cancel");
        });

        Self { latest, wake }
    }

    /// Test-only: create a publisher with no drain task, exposing the
    /// coalescing map for assertions. `notify()` still updates the map so
    /// callers can observe which addresses/CIDs would have been published.
    #[cfg(test)]
    #[allow(clippy::type_complexity)]
    pub fn inert_for_test() -> (Self, Arc<Mutex<HashMap<String, [u8; CID_LEN]>>>) {
        let latest: Arc<Mutex<HashMap<String, [u8; CID_LEN]>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let wake = Arc::new(Notify::new());
        (
            Self {
                latest: Arc::clone(&latest),
                wake,
            },
            latest,
        )
    }

    /// Announce a new root CID for a user.
    ///
    /// Overwrites any previous pending CID for the same address and wakes the
    /// drain task. Callable from sync context.
    pub fn notify(&self, addr_hex: String, root_cid: [u8; CID_LEN]) {
        {
            let mut map = self
                .latest
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            map.insert(addr_hex, root_cid);
        }
        self.wake.notify_one();
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

        let (publisher, latest) = ZenohPublisher::inert_for_test();

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

        // notify() is synchronous: the coalescing map is updated before
        // insert_message returns, so no polling or timeout is needed.
        let map = latest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let expected_addr_hex = hex::encode(addr);
        let cid = map
            .get(&expected_addr_hex)
            .expect("ZenohPublisher did not record a notification");
        assert_eq!(*cid, *mgr.get_root(&addr).unwrap());
        assert_eq!(map.len(), 1);
    }

    #[tokio::test]
    async fn zenoh_publisher_coalesces_per_addr() {
        // Two bursts of notifications for the same addr must collapse to the
        // latest CID, and the drain task must snapshot the newest value.
        let (publisher, latest) = ZenohPublisher::inert_for_test();

        let addr_hex = "aa".repeat(ADDRESS_HASH_LEN);
        let cid_old = [0x01u8; CID_LEN];
        let cid_mid = [0x02u8; CID_LEN];
        let cid_new = [0x03u8; CID_LEN];

        publisher.notify(addr_hex.clone(), cid_old);
        publisher.notify(addr_hex.clone(), cid_mid);
        publisher.notify(addr_hex.clone(), cid_new);

        let map = latest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(map.len(), 1, "coalescing must keep one entry per addr");
        assert_eq!(
            map.get(&addr_hex).copied(),
            Some(cid_new),
            "latest CID must win; older values must not linger"
        );
    }

    #[tokio::test]
    async fn zenoh_publisher_keeps_distinct_addrs_separate() {
        // Updates for different addrs must not clobber each other.
        let (publisher, latest) = ZenohPublisher::inert_for_test();

        let alice_hex = "aa".repeat(ADDRESS_HASH_LEN);
        let bob_hex = "bb".repeat(ADDRESS_HASH_LEN);
        let alice_cid = [0x11u8; CID_LEN];
        let bob_cid = [0x22u8; CID_LEN];

        publisher.notify(alice_hex.clone(), alice_cid);
        publisher.notify(bob_hex.clone(), bob_cid);

        let map = latest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&alice_hex).copied(), Some(alice_cid));
        assert_eq!(map.get(&bob_hex).copied(), Some(bob_cid));
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

    /// Exercises the REAL `ZenohPublisher::new()` drain task (not the mpsc
    /// test double) end-to-end: opens two in-process Zenoh sessions in peer
    /// mode over loopback, creates the publisher with one session, declares
    /// a subscriber on the other, inserts a message, and verifies the raw
    /// root CID bytes reach the subscriber through `session.put()`.
    ///
    /// This catches regressions the `from_sender` test cannot — e.g., if the
    /// drain task stops spawning, if the topic format breaks, or if
    /// `session.put()` starts returning errors silently.
    ///
    /// `#[ignore]`'d by default: starts two Zenoh sessions which bind sockets
    /// and rely on peer discovery; can be flaky under loaded CI runners.
    /// Run locally with `cargo test -p harmony-mail -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn mailbox_manager_real_zenoh_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        // Two peer-mode sessions that discover each other on localhost.
        let pub_session = zenoh::open(zenoh::Config::default()).await.unwrap();
        let sub_session = zenoh::open(zenoh::Config::default()).await.unwrap();

        let addr = [0xCCu8; ADDRESS_HASH_LEN];
        let addr_hex = hex::encode(addr);
        let topic = format!("harmony/mail/v1/{addr_hex}/root");

        let subscriber = sub_session.declare_subscriber(&topic).await.unwrap();

        // Brief settle time for peer discovery on loopback.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let cancel = CancellationToken::new();
        let publisher = ZenohPublisher::new(pub_session, cancel.clone());

        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        mgr.set_publisher(publisher);
        mgr.ensure_user_mailbox(&addr).unwrap();

        mgr.insert_message(
            &addr,
            &dummy_msg_cid(7),
            &dummy_msg_id(7),
            &dummy_sender(),
            1700000042,
            "Real Zenoh",
        )
        .unwrap();

        // The drain task must pick up the coalesced update and run
        // session.put().
        let sample = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            subscriber.recv_async(),
        )
        .await
        .expect("subscriber never received the real zenoh publish within 5s")
        .expect("subscriber channel closed");

        let payload = sample.payload().to_bytes();
        assert_eq!(payload.len(), CID_LEN);
        let received_cid: [u8; CID_LEN] = (&payload[..]).try_into().unwrap();
        assert_eq!(received_cid, *mgr.get_root(&addr).unwrap());

        // Signal shutdown and give the drain task a chance to exit cleanly.
        cancel.cancel();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    /// Regression test for the drain-task shutdown path.
    ///
    /// The publisher's drain task must observe the shared `CancellationToken`
    /// so graceful shutdown can tear it down deterministically instead of
    /// leaking it until runtime drop. This test opens a real Zenoh session
    /// (so we exercise the real `new` path, not the `inert_for_test` scaffold),
    /// cancels the token, and confirms a subsequent `notify()` produces no
    /// publish — i.e., the drain task has stopped consuming wake-ups.
    ///
    /// `#[ignore]`'d for the same reason as the end-to-end test — relies on
    /// peer discovery over loopback.
    #[tokio::test]
    #[ignore]
    async fn zenoh_publisher_drain_task_stops_on_cancel() {
        let pub_session = zenoh::open(zenoh::Config::default()).await.unwrap();
        let sub_session = zenoh::open(zenoh::Config::default()).await.unwrap();

        let addr_hex = "dd".repeat(ADDRESS_HASH_LEN);
        let topic = format!("harmony/mail/v1/{addr_hex}/root");
        let subscriber = sub_session.declare_subscriber(&topic).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let cancel = CancellationToken::new();
        let publisher = ZenohPublisher::new(pub_session, cancel.clone());

        // Sanity: the drain task is alive and publishes before cancel.
        publisher.notify(addr_hex.clone(), [0x01u8; CID_LEN]);
        let before = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            subscriber.recv_async(),
        )
        .await
        .expect("pre-cancel publish should succeed")
        .expect("subscriber channel closed");
        assert_eq!(before.payload().to_bytes().len(), CID_LEN);

        // Cancel and give the drain task a moment to exit.
        cancel.cancel();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // A post-cancel notify should wake nothing — no further publish
        // arrives within a short window.
        publisher.notify(addr_hex.clone(), [0x02u8; CID_LEN]);
        let after = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            subscriber.recv_async(),
        )
        .await;
        assert!(
            after.is_err(),
            "drain task should have stopped after cancel; got unexpected publish"
        );
    }
}

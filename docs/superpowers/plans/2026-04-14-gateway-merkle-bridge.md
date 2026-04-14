# Gateway Merkle Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dual-write inbound SMTP messages to the CAS Merkle mailbox alongside the existing ImapStore, with per-user MailRoot trees and Zenoh root CID publication.

**Architecture:** A new `MailboxManager` holds per-user Merkle tree root CIDs in memory, backed by a small SQLite table for persistence. On SMTP delivery, Phase 5 updates the Merkle tree (page → folder → root) via CAS, then publishes the new root CID to Zenoh. The IMAP store is unmodified — both write paths are independent.

**Tech Stack:** Rust, rusqlite, harmony-content (CAS), harmony-db (DiskBookStore), Zenoh (pub/sub + queryable)

---

### Task 1: Add `list_users` to ImapStore

**Files:**
- Modify: `crates/harmony-mail/src/imap_store.rs`

The MailboxManager needs to enumerate all registered users at startup to initialize their Merkle trees. The ImapStore has no `list_users` method yet.

- [ ] **Step 1: Write the test**

Add to the existing `imap_store::tests` module in `crates/harmony-mail/src/imap_store.rs`:

```rust
#[test]
fn list_users_returns_all() {
    let dir = tempfile::tempdir().unwrap();
    let store = ImapStore::open(&dir.path().join("test.db")).unwrap();
    store.initialize_default_mailboxes().unwrap();

    let addr1 = [0xAAu8; ADDRESS_HASH_LEN];
    let addr2 = [0xBBu8; ADDRESS_HASH_LEN];
    store.create_user("alice", "pass1", &addr1).unwrap();
    store.create_user("bob", "pass2", &addr2).unwrap();

    let users = store.list_users().unwrap();
    assert_eq!(users.len(), 2);
    let usernames: Vec<&str> = users.iter().map(|u| u.username.as_str()).collect();
    assert!(usernames.contains(&"alice"));
    assert!(usernames.contains(&"bob"));
}

#[test]
fn list_users_empty() {
    let dir = tempfile::tempdir().unwrap();
    let store = ImapStore::open(&dir.path().join("test.db")).unwrap();
    store.initialize_default_mailboxes().unwrap();
    let users = store.list_users().unwrap();
    assert!(users.is_empty());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-mail list_users -- --nocapture`
Expected: FAIL — `list_users` method does not exist.

- [ ] **Step 3: Implement `list_users`**

Add to the `impl ImapStore` block in `crates/harmony-mail/src/imap_store.rs`, near the existing `get_user` and `get_user_by_address` methods:

```rust
/// List all registered users.
pub fn list_users(&self) -> Result<Vec<UserRow>, StoreError> {
    let conn = self.conn.lock().unwrap();
    let mut stmt = conn.prepare(
        "SELECT id, username, harmony_address, created_at FROM users ORDER BY id",
    )?;
    let rows = stmt.query_map([], |row| {
        let harmony_blob: Vec<u8> = row.get(2)?;
        let mut harmony_address = [0u8; ADDRESS_HASH_LEN];
        if harmony_blob.len() == ADDRESS_HASH_LEN {
            harmony_address.copy_from_slice(&harmony_blob);
        }
        Ok(UserRow {
            id: row.get(0)?,
            username: row.get(1)?,
            harmony_address,
            created_at: row.get(3)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(StoreError::from)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-mail list_users -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/imap_store.rs
git commit -m "feat(mail): add list_users to ImapStore for Merkle mailbox init"
```

---

### Task 2: Create `MailboxManager` with root CID persistence

**Files:**
- Create: `crates/harmony-mail/src/mailbox_manager.rs`
- Modify: `crates/harmony-mail/src/lib.rs`

This task creates the `MailboxManager` struct with SQLite-backed root CID persistence, empty mailbox initialization, and the `list_roots` method. No CAS operations yet — just the persistence layer.

- [ ] **Step 1: Write the tests**

Create `crates/harmony-mail/src/mailbox_manager.rs` with tests at the bottom:

```rust
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

        // Load existing roots into memory
        let mut roots = HashMap::new();
        let mut stmt = db.prepare("SELECT address, root_cid FROM mailbox_roots")?;
        let rows = stmt.query_map([], |row| {
            let addr_blob: Vec<u8> = row.get(0)?;
            let cid_blob: Vec<u8> = row.get(1)?;
            Ok((addr_blob, cid_blob))
        })?;
        for row in rows {
            let (addr_blob, cid_blob) = row?;
            if addr_blob.len() == ADDRESS_HASH_LEN && cid_blob.len() == CID_LEN {
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
```

- [ ] **Step 2: Register the module**

Add to `crates/harmony-mail/src/lib.rs`:

```rust
pub mod mailbox_manager;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p harmony-mail mailbox_manager -- --nocapture`
Expected: PASS (3 tests)

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-mail/src/mailbox_manager.rs crates/harmony-mail/src/lib.rs
git commit -m "feat(mail): add MailboxManager with root CID persistence"
```

---

### Task 3: Initialize empty Merkle trees for users

**Files:**
- Modify: `crates/harmony-mail/src/mailbox_manager.rs`

Add the `ensure_user_mailbox` method that creates an empty MailRoot with 4 empty folders and pages in CAS, and persists the root CID. This is called at startup for each user that doesn't already have a Merkle tree.

- [ ] **Step 1: Write the test**

Add to `mailbox_manager::tests`:

```rust
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-mail ensure_user_mailbox -- --nocapture`
Expected: FAIL — method does not exist.

- [ ] **Step 3: Implement `ensure_user_mailbox`**

Add a helper to ingest bytes into CAS and the `ensure_user_mailbox` method to `impl MailboxManager`:

```rust
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
fn cas_load(&self, cid: &[u8; CID_LEN]) -> Result<Vec<u8>, MailboxError> {
    let book = harmony_db::DiskBookStore::new(&self.content_store_path);
    let content_id = harmony_content::cid::ContentId::from_bytes(*cid);
    book.get(&content_id)
        .map(|b| b.to_vec())
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-mail mailbox_manager -- --nocapture`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/mailbox_manager.rs
git commit -m "feat(mail): MailboxManager creates empty Merkle trees for users"
```

---

### Task 4: Implement `insert_message` for Merkle mailbox

**Files:**
- Modify: `crates/harmony-mail/src/mailbox_manager.rs`

The core operation: load the inbox tree from CAS, prepend a new MessageEntry to the head page (splitting if full), rebuild folder and root, write back to CAS, update persistence.

- [ ] **Step 1: Write the tests**

Add to `mailbox_manager::tests`:

```rust
use crate::mailbox::MessageEntry;

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

    // Verify: 1 page, 100 entries
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
    for i in 0..3 {
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-mail insert_into_empty -- --nocapture`
Expected: FAIL — `insert_message` method does not exist.

- [ ] **Step 3: Implement `insert_message`**

Add to `impl MailboxManager`:

```rust
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
) -> Result<(), MailboxError> {
    let root_cid = *self
        .roots
        .get(user_address)
        .ok_or(MailboxError::NoMailbox(*user_address))?;

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

    Ok(())
}
```

Note: `truncate_utf8` in `mailbox.rs` is currently private. Change it to `pub(crate)`:

In `crates/harmony-mail/src/mailbox.rs`, change:
```rust
fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
```
to:
```rust
pub(crate) fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-mail mailbox_manager -- --nocapture`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/mailbox_manager.rs crates/harmony-mail/src/mailbox.rs
git commit -m "feat(mail): MailboxManager.insert_message with page splitting"
```

---

### Task 5: Add ZenohConfig to gateway configuration

**Files:**
- Modify: `crates/harmony-mail/src/config.rs`

Add an optional `[zenoh]` section to the config. The gateway works without it — Zenoh is an optional enhancement.

- [ ] **Step 1: Write the test**

Add to `config::tests` in `crates/harmony-mail/src/config.rs`:

```rust
#[test]
fn parse_config_with_zenoh() {
    let config_str = format!(
        "{}\n\n[zenoh]\nenabled = true\nendpoint = \"tcp/192.168.1.1:7447\"\n",
        FULL_CONFIG
    );
    let config = Config::from_toml(&config_str).expect("should parse with zenoh section");
    let zenoh = config.zenoh.expect("zenoh should be Some");
    assert!(zenoh.enabled);
    assert_eq!(zenoh.endpoint.as_deref(), Some("tcp/192.168.1.1:7447"));
}

#[test]
fn parse_config_without_zenoh() {
    let config = Config::from_toml(FULL_CONFIG).expect("should parse without zenoh section");
    assert!(config.zenoh.is_none());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-mail parse_config_with_zenoh -- --nocapture`
Expected: FAIL — `zenoh` field does not exist on Config.

- [ ] **Step 3: Add ZenohConfig**

In `crates/harmony-mail/src/config.rs`, add the struct and the field on `Config`:

```rust
/// Optional Zenoh configuration for Merkle mailbox publication.
///
/// If absent or disabled, SMTP delivery still works — messages go to the
/// IMAP store and CAS, but root CID updates are not published to the network.
#[derive(Debug, Deserialize)]
pub struct ZenohConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Explicit Zenoh endpoint. If omitted, uses Zenoh's default peer discovery.
    pub endpoint: Option<String>,
}
```

Add to the `Config` struct:

```rust
#[serde(default)]
pub zenoh: Option<ZenohConfig>,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-mail config -- --nocapture`
Expected: PASS (all config tests)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/config.rs
git commit -m "feat(mail): add optional ZenohConfig for Merkle mailbox publication"
```

---

### Task 6: Wire MailboxManager into gateway startup and SMTP delivery

**Files:**
- Modify: `crates/harmony-mail/src/server.rs`

This is the integration task. Create the MailboxManager in `run()`, thread it through SMTP handlers, and add Phase 5 to `DeliverToHarmony`. Also change Phase 4 to track which recipients were delivered (not just a count).

- [ ] **Step 1: Write the integration test**

Add to `server::tests` in `crates/harmony-mail/src/server.rs`:

```rust
#[tokio::test]
async fn smtp_delivers_to_merkle_mailbox() {
    let config = test_config();
    let max_message_size = parse_message_size(&config.spam.max_message_size);
    let smtp_config = SmtpConfig {
        domain: config.domain.name.clone(),
        mx_host: config.domain.mx_host.clone(),
        max_message_size,
        max_recipients: 100,
        tls_available: false,
    };

    let test_dir = tempfile::tempdir().unwrap();
    let store = Arc::new(
        crate::imap_store::ImapStore::open(&test_dir.path().join("imap.db")).unwrap(),
    );
    store.initialize_default_mailboxes().unwrap();
    let alice_addr = [0xAAu8; crate::message::ADDRESS_HASH_LEN];
    store.create_user("alice", "alicepass", &alice_addr).unwrap();

    let content_path = test_dir.path().join("content");
    std::fs::create_dir_all(content_path.join("commits")).unwrap();
    std::fs::create_dir_all(content_path.join("blobs")).unwrap();

    // Create MailboxManager and init alice's tree
    let mailbox_mgr = Arc::new(std::sync::Mutex::new(
        crate::mailbox_manager::MailboxManager::open(
            &test_dir.path().join("mailbox_roots.db"),
            &content_path,
        )
        .unwrap(),
    ));
    mailbox_mgr
        .lock()
        .unwrap()
        .ensure_user_mailbox(&alice_addr)
        .unwrap();

    let store_clone = store.clone();
    let content_path_clone = content_path.clone();
    let mgr_clone = Some(Arc::clone(&mailbox_mgr));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server_handle = tokio::spawn(async move {
        let (stream, peer_addr) = listener.accept().await.unwrap();
        handle_connection(
            stream,
            peer_addr.ip(),
            smtp_config,
            max_message_size,
            None,
            store_clone,
            None,
            "test.example.com".to_string(),
            content_path_clone,
            5,
            mgr_clone,
        )
        .await
        .unwrap();
    });

    let stream = TcpStream::connect(addr).await.unwrap();
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    // SMTP conversation
    let greeting = read_smtp_response(&mut reader).await;
    assert!(greeting.starts_with("220 "), "greeting: {greeting}");

    write_half.write_all(b"EHLO sender.test.com\r\n").await.unwrap();
    let _ = read_smtp_response(&mut reader).await;

    write_half.write_all(b"MAIL FROM:<sender@test.com>\r\n").await.unwrap();
    let _ = read_smtp_response(&mut reader).await;

    write_half.write_all(b"RCPT TO:<alice@test.example.com>\r\n").await.unwrap();
    let _ = read_smtp_response(&mut reader).await;

    write_half.write_all(b"DATA\r\n").await.unwrap();
    let _ = read_smtp_response(&mut reader).await;

    write_half
        .write_all(
            b"From: sender@test.com\r\n\
              To: alice@test.example.com\r\n\
              Subject: Merkle Bridge Test\r\n\
              Message-ID: <merkle-test-001@test.com>\r\n\
              Date: Mon, 14 Apr 2026 00:00:00 +0000\r\n\
              \r\n\
              Hello Alice, testing dual-write.\r\n\
              .\r\n",
        )
        .await
        .unwrap();
    let deliver_resp = read_smtp_response(&mut reader).await;
    assert!(deliver_resp.contains("250"), "delivery: {deliver_resp}");

    write_half.write_all(b"QUIT\r\n").await.unwrap();
    let _ = read_smtp_response(&mut reader).await;
    server_handle.await.unwrap();

    // Verify IMAP store has the message
    let mbox = store.get_mailbox("INBOX").unwrap().unwrap();
    let imap_messages = store.get_messages(mbox.id).unwrap();
    assert_eq!(imap_messages.len(), 1);
    let imap_cid = imap_messages[0].message_cid.expect("IMAP msg should have CID");

    // Verify Merkle mailbox has the message
    let mgr = mailbox_mgr.lock().unwrap();
    let root_cid = mgr.get_root(&alice_addr).expect("alice should have a root");
    let book = harmony_db::DiskBookStore::new(&content_path);
    let root = crate::mailbox::MailRoot::from_bytes(
        book.get(&harmony_content::cid::ContentId::from_bytes(*root_cid)).unwrap(),
    )
    .unwrap();
    let folder = crate::mailbox::MailFolder::from_bytes(
        book.get(&harmony_content::cid::ContentId::from_bytes(
            *root.folder_cid(crate::mailbox::FolderKind::Inbox),
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(folder.message_count, 1);

    let page = crate::mailbox::MailPage::from_bytes(
        book.get(&harmony_content::cid::ContentId::from_bytes(folder.page_cids[0])).unwrap(),
    )
    .unwrap();
    assert_eq!(page.entries.len(), 1);
    assert_eq!(page.entries[0].subject_snippet, "Merkle Bridge Test");

    // Dual-write consistency: both stores reference the same CAS message CID
    assert_eq!(page.entries[0].message_cid, imap_cid);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-mail smtp_delivers_to_merkle -- --nocapture`
Expected: FAIL — `handle_connection` doesn't accept `mailbox_manager` parameter.

- [ ] **Step 3: Wire MailboxManager through the gateway**

**3a. Add the `mailbox_manager` parameter to handler functions.**

In `crates/harmony-mail/src/server.rs`, update `handle_connection` signature to add:

```rust
mailbox_manager: Option<Arc<std::sync::Mutex<crate::mailbox_manager::MailboxManager>>>,
```

Thread it through to `handle_connection_generic` and `process_async_actions` the same way `imap_store` is threaded. Add the parameter to `process_async_actions`:

```rust
mailbox_manager: &Option<Arc<std::sync::Mutex<crate::mailbox_manager::MailboxManager>>>,
```

**3b. Extract sender_address and subject before Phase 3 closure.**

In `DeliverToHarmony`, after translation (Phase 2) but before the Phase 3 closure, add:

```rust
let sender_address = translated.message.sender_address;
let subject_snippet = crate::mailbox::truncate_utf8(
    &translated.message.subject,
    crate::mailbox::MAX_SNIPPET_LEN,
)
.to_string();
```

**3c. Change Phase 4 to track delivered addresses.**

Replace `let mut delivered_count = 0u32;` with `let mut delivered_to: Vec<[u8; crate::message::ADDRESS_HASH_LEN]> = Vec::new();` and change `delivered_count += 1` to `delivered_to.push(*recipient_hash);`. Change `let success = delivered_count > 0;` to `let success = !delivered_to.is_empty();`.

**3d. Add Phase 5 after Phase 4.**

After the `let success = ...` line and before `let callback_actions = session.handle(...)`:

```rust
// Phase 5: Update Merkle mailbox (non-critical)
if let Some(ref mgr) = mailbox_manager {
    if let Some(ref msg_cid) = message_cid {
        for addr in &delivered_to {
            if let Err(e) = mgr.lock().unwrap().insert_message(
                addr,
                msg_cid,
                &msg_id,
                &sender_address,
                timestamp,
                &subject_snippet,
            ) {
                tracing::warn!(
                    recipient = hex::encode(addr),
                    error = %e,
                    "Merkle mailbox update failed, IMAP delivery unaffected"
                );
            }
        }
    }
}
```

**3e. Create MailboxManager in `run()`.**

After the ImapStore and CAS directory setup, before the cancellation token:

```rust
// ── Merkle mailbox manager ────────────────────────────────────────
let store_dir = Path::new(&config.imap.store_path)
    .parent()
    .unwrap_or(Path::new("."));
let mailbox_mgr = match crate::mailbox_manager::MailboxManager::open(
    &store_dir.join("mailbox_roots.db"),
    &content_store_path,
) {
    Ok(mut mgr) => {
        // Initialize Merkle trees for all registered users
        for user in imap_store.list_users().unwrap_or_default() {
            if let Err(e) = mgr.ensure_user_mailbox(&user.harmony_address) {
                tracing::warn!(
                    user = %user.username,
                    error = %e,
                    "failed to init Merkle mailbox"
                );
            }
        }
        tracing::info!(
            users = mgr.roots.len(),
            "Merkle mailbox manager initialized"
        );
        Some(Arc::new(std::sync::Mutex::new(mgr)))
    }
    Err(e) => {
        tracing::warn!(error = %e, "failed to open Merkle mailbox manager, Merkle writes disabled");
        None
    }
};
```

Note: `mgr.roots` is private. Either make it `pub(crate)` or add a `pub fn user_count(&self) -> usize` method returning `self.roots.len()`.

**3f. Thread `mailbox_mgr` through all SMTP listener spawn sites.**

Clone and pass `mailbox_mgr` to each `handle_connection` call in the port 25, 465, and 587 listener blocks, the same way `imap_store` is cloned and passed.

**3g. Update existing test call sites.**

All existing calls to `handle_connection` in tests need the new `mailbox_manager: None` parameter appended (for tests that don't need Merkle writes).

- [ ] **Step 4: Run all tests**

Run: `cargo test -p harmony-mail -- --nocapture`
Expected: PASS (all existing tests + new `smtp_delivers_to_merkle_mailbox`)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/server.rs crates/harmony-mail/src/mailbox_manager.rs
git commit -m "feat(mail): wire MailboxManager into SMTP delivery (Phase 5 dual-write)"
```

---

### Task 7: Root CID persistence across restart

**Files:**
- Modify: `crates/harmony-mail/src/mailbox_manager.rs`

Add a test verifying that the full cycle works: create manager, insert messages, drop, reopen from disk, verify the tree is intact.

- [ ] **Step 1: Write the test**

Add to `mailbox_manager::tests`:

```rust
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
        for i in 0..3 {
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
```

- [ ] **Step 2: Run test**

Run: `cargo test -p harmony-mail root_cid_survives -- --nocapture`
Expected: PASS (this tests existing functionality — if it fails, there's a bug to fix)

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail/src/mailbox_manager.rs
git commit -m "test(mail): verify Merkle mailbox root CID persistence across restart"
```

---

### Note: Zenoh Publication (Deferred)

The spec calls for Zenoh pub/sub + queryable to publish root CID updates. Adding `zenoh` as a dependency is a significant change (large transitive dependency tree) and the integration should be verified against the harmony-zenoh crate's actual APIs. The Merkle write path to CAS is the core value of this plan — messages are durably stored and discoverable by CID. Zenoh publication is a thin layer on top that can be added as a follow-up once the CAS foundation is solid and tested.

When ready, the additions are:
- Add `zenoh` to Cargo.toml dependencies
- Add `zenoh: Option<Arc<zenoh::Session>>` field to MailboxManager
- At end of `insert_message`, publish root CID bytes to `harmony/messages/{addr_hex}/inbox`
- At startup, register a queryable per user that returns current root CID from HashMap

---

### Task 8: Run full test suite and verify

**Files:** None (verification only)

- [ ] **Step 1: Run all harmony-mail tests**

Run: `cargo test -p harmony-mail -- --nocapture`
Expected: All tests pass (383+ existing + new tests)

- [ ] **Step 2: Build full workspace**

Run: `cargo build --workspace`
Expected: Clean build, no errors

- [ ] **Step 3: Verify test count increased**

Count the total: should be 383 (baseline from PR #228) + new tests from this plan.

//! IMAP mailbox store backed by SQLite.
//!
//! Manages UID assignment, flags, mailbox metadata, and user credentials.
//! The source of truth for message *content* is always Harmony-native; this
//! store only maintains the IMAP-specific view (UIDs, flags, sequence).

use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::message::{ADDRESS_HASH_LEN, MESSAGE_ID_LEN};

// ── Schema version ──────────────────────────────────────────────────

const SCHEMA_VERSION: u32 = 1;

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS mailboxes (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    uid_validity INTEGER NOT NULL,
    uid_next     INTEGER NOT NULL DEFAULT 1,
    subscribed   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id             INTEGER PRIMARY KEY,
    mailbox_id     INTEGER NOT NULL REFERENCES mailboxes(id) ON DELETE CASCADE,
    uid            INTEGER NOT NULL,
    harmony_msg_id BLOB NOT NULL,
    internal_date  INTEGER NOT NULL,
    rfc822_size    INTEGER NOT NULL DEFAULT 0,
    UNIQUE(mailbox_id, uid)
);

CREATE TABLE IF NOT EXISTS flags (
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    flag       TEXT NOT NULL,
    UNIQUE(message_id, flag)
);

CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY,
    username        TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    harmony_address BLOB NOT NULL,
    created_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_mailbox_uid ON messages(mailbox_id, uid);
CREATE INDEX IF NOT EXISTS idx_messages_harmony_id ON messages(harmony_msg_id);
CREATE INDEX IF NOT EXISTS idx_flags_message ON flags(message_id);
";

const DEFAULT_MAILBOXES: &[&str] = &["INBOX", "Sent", "Drafts", "Trash", "Junk"];

// ── Error type ──────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("mailbox not found: {0}")]
    MailboxNotFound(String),
    #[error("mailbox already exists: {0}")]
    MailboxExists(String),
    #[error("message not found")]
    MessageNotFound,
    #[error("user not found: {0}")]
    UserNotFound(String),
    #[error("user already exists: {0}")]
    UserExists(String),
    #[error("authentication failed")]
    AuthFailed,
    #[error("password hashing error: {0}")]
    HashError(String),
}

// ── Row types ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MailboxRow {
    pub id: i64,
    pub name: String,
    pub uid_validity: u32,
    pub uid_next: u32,
    pub subscribed: bool,
}

#[derive(Debug, Clone)]
pub struct MessageRow {
    pub id: i64,
    pub mailbox_id: i64,
    pub uid: u32,
    pub harmony_msg_id: [u8; MESSAGE_ID_LEN],
    pub internal_date: u64,
    pub rfc822_size: u32,
}

#[derive(Debug, Clone)]
pub struct UserRow {
    pub id: i64,
    pub username: String,
    pub harmony_address: [u8; ADDRESS_HASH_LEN],
    pub created_at: u64,
}

// ── Store ───────────────────────────────────────────────────────────

/// Thread-safe IMAP store. The rusqlite::Connection is wrapped in a Mutex
/// because it is `!Send`. All methods acquire the lock for the duration of
/// their operation.
pub struct ImapStore {
    conn: Mutex<rusqlite::Connection>,
}

// Safety: The Mutex serializes all access to the !Send rusqlite::Connection.
unsafe impl Send for ImapStore {}
unsafe impl Sync for ImapStore {}

impl ImapStore {
    /// Open or create the IMAP store at the given path.
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        let conn = if path.as_os_str() == ":memory:" {
            rusqlite::Connection::open_in_memory()?
        } else {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    StoreError::Sqlite(rusqlite::Error::SqliteFailure(
                        rusqlite::ffi::Error::new(1),
                        Some(format!("cannot create directory: {e}")),
                    ))
                })?;
            }
            rusqlite::Connection::open(path)?
        };
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.migrate()?;
        Ok(store)
    }

    /// Open an in-memory store (for testing).
    pub fn open_memory() -> Result<Self, StoreError> {
        Self::open(Path::new(":memory:"))
    }

    fn migrate(&self) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let version: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);
        if version < SCHEMA_VERSION {
            conn.execute_batch(SCHEMA_SQL)?;
            conn.execute("DELETE FROM schema_version", [])?;
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?1)",
                [SCHEMA_VERSION],
            )?;
        }
        Ok(())
    }

    /// Create default mailboxes if they don't already exist.
    pub fn initialize_default_mailboxes(&self) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let uid_validity = now_secs() as u32;
        for name in DEFAULT_MAILBOXES {
            conn.execute(
                "INSERT OR IGNORE INTO mailboxes (name, uid_validity) VALUES (?1, ?2)",
                rusqlite::params![name, uid_validity],
            )?;
        }
        Ok(())
    }

    // ── Mailbox operations ──────────────────────────────────────────

    pub fn get_mailbox(&self, name: &str) -> Result<Option<MailboxRow>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, uid_validity, uid_next, subscribed FROM mailboxes WHERE name = ?1",
        )?;
        let row = stmt
            .query_row([name], |row| {
                Ok(MailboxRow {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    uid_validity: row.get(2)?,
                    uid_next: row.get(3)?,
                    subscribed: row.get::<_, i32>(4)? != 0,
                })
            })
            .optional()?;
        Ok(row)
    }

    pub fn create_mailbox(&self, name: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let uid_validity = now_secs() as u32;
        match conn.execute(
            "INSERT INTO mailboxes (name, uid_validity) VALUES (?1, ?2)",
            rusqlite::params![name, uid_validity],
        ) {
            Ok(_) => Ok(()),
            Err(rusqlite::Error::SqliteFailure(err, _))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Err(StoreError::MailboxExists(name.to_string()))
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn delete_mailbox(&self, name: &str) -> Result<(), StoreError> {
        if name == "INBOX" {
            return Err(StoreError::MailboxNotFound(
                "cannot delete INBOX".to_string(),
            ));
        }
        let conn = self.conn.lock().unwrap();
        let affected = conn.execute("DELETE FROM mailboxes WHERE name = ?1", [name])?;
        if affected == 0 {
            return Err(StoreError::MailboxNotFound(name.to_string()));
        }
        Ok(())
    }

    pub fn list_mailboxes(&self, pattern: &str) -> Result<Vec<MailboxRow>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let sql_pattern = imap_pattern_to_sql(pattern);
        let mut stmt = conn.prepare(
            "SELECT id, name, uid_validity, uid_next, subscribed FROM mailboxes WHERE name LIKE ?1 ORDER BY name",
        )?;
        let rows = stmt
            .query_map([&sql_pattern], |row| {
                Ok(MailboxRow {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    uid_validity: row.get(2)?,
                    uid_next: row.get(3)?,
                    subscribed: row.get::<_, i32>(4)? != 0,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn subscribe(&self, name: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let affected = conn.execute(
            "UPDATE mailboxes SET subscribed = 1 WHERE name = ?1",
            [name],
        )?;
        if affected == 0 {
            return Err(StoreError::MailboxNotFound(name.to_string()));
        }
        Ok(())
    }

    pub fn unsubscribe(&self, name: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let affected = conn.execute(
            "UPDATE mailboxes SET subscribed = 0 WHERE name = ?1",
            [name],
        )?;
        if affected == 0 {
            return Err(StoreError::MailboxNotFound(name.to_string()));
        }
        Ok(())
    }

    // ── Message operations ──────────────────────────────────────────

    pub fn insert_message(
        &self,
        mailbox_name: &str,
        harmony_msg_id: &[u8; MESSAGE_ID_LEN],
        internal_date: u64,
        rfc822_size: u32,
    ) -> Result<u32, StoreError> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;

        let (mailbox_id, uid_next): (i64, u32) = tx
            .query_row(
                "SELECT id, uid_next FROM mailboxes WHERE name = ?1",
                [mailbox_name],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|_| StoreError::MailboxNotFound(mailbox_name.to_string()))?;

        let uid = uid_next;
        tx.execute(
            "UPDATE mailboxes SET uid_next = ?1 WHERE id = ?2",
            rusqlite::params![uid + 1, mailbox_id],
        )?;
        tx.execute(
            "INSERT INTO messages (mailbox_id, uid, harmony_msg_id, internal_date, rfc822_size) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![mailbox_id, uid, harmony_msg_id.as_slice(), internal_date, rfc822_size],
        )?;
        tx.commit()?;
        Ok(uid)
    }

    pub fn get_messages(&self, mailbox_id: i64) -> Result<Vec<MessageRow>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, mailbox_id, uid, harmony_msg_id, internal_date, rfc822_size FROM messages WHERE mailbox_id = ?1 ORDER BY uid",
        )?;
        let rows = stmt
            .query_map([mailbox_id], row_to_message)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn get_message_by_uid(
        &self,
        mailbox_id: i64,
        uid: u32,
    ) -> Result<Option<MessageRow>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, mailbox_id, uid, harmony_msg_id, internal_date, rfc822_size FROM messages WHERE mailbox_id = ?1 AND uid = ?2",
        )?;
        let row = stmt
            .query_row(rusqlite::params![mailbox_id, uid], row_to_message)
            .optional()?;
        Ok(row)
    }

    pub fn get_messages_by_uid_range(
        &self,
        mailbox_id: i64,
        uid_start: u32,
        uid_end: u32,
    ) -> Result<Vec<MessageRow>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, mailbox_id, uid, harmony_msg_id, internal_date, rfc822_size FROM messages WHERE mailbox_id = ?1 AND uid >= ?2 AND uid <= ?3 ORDER BY uid",
        )?;
        let rows = stmt
            .query_map(
                rusqlite::params![mailbox_id, uid_start, uid_end],
                row_to_message,
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn count_messages(&self, mailbox_id: i64) -> Result<u32, StoreError> {
        let conn = self.conn.lock().unwrap();
        let count: u32 = conn.query_row(
            "SELECT COUNT(*) FROM messages WHERE mailbox_id = ?1",
            [mailbox_id],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    pub fn count_recent(&self, mailbox_id: i64) -> Result<u32, StoreError> {
        let conn = self.conn.lock().unwrap();
        let count: u32 = conn.query_row(
            "SELECT COUNT(*) FROM messages m JOIN flags f ON m.id = f.message_id WHERE m.mailbox_id = ?1 AND f.flag = '\\Recent'",
            [mailbox_id], |row| row.get(0),
        )?;
        Ok(count)
    }

    pub fn count_unseen(&self, mailbox_id: i64) -> Result<u32, StoreError> {
        let conn = self.conn.lock().unwrap();
        let count: u32 = conn.query_row(
            "SELECT COUNT(*) FROM messages m WHERE m.mailbox_id = ?1 AND m.id NOT IN (SELECT message_id FROM flags WHERE flag = '\\Seen')",
            [mailbox_id], |row| row.get(0),
        )?;
        Ok(count)
    }

    // ── Flag operations ─────────────────────────────────────────────

    pub fn get_flags(&self, message_db_id: i64) -> Result<Vec<String>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT flag FROM flags WHERE message_id = ?1 ORDER BY flag")?;
        let flags = stmt
            .query_map([message_db_id], |row| row.get(0))?
            .collect::<Result<Vec<String>, _>>()?;
        Ok(flags)
    }

    pub fn set_flags(&self, message_db_id: i64, flags: &[&str]) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        tx.execute("DELETE FROM flags WHERE message_id = ?1", [message_db_id])?;
        for flag in flags {
            tx.execute(
                "INSERT INTO flags (message_id, flag) VALUES (?1, ?2)",
                rusqlite::params![message_db_id, flag],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn add_flags(&self, message_db_id: i64, flags: &[&str]) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        for flag in flags {
            conn.execute(
                "INSERT OR IGNORE INTO flags (message_id, flag) VALUES (?1, ?2)",
                rusqlite::params![message_db_id, flag],
            )?;
        }
        Ok(())
    }

    pub fn remove_flags(&self, message_db_id: i64, flags: &[&str]) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        for flag in flags {
            conn.execute(
                "DELETE FROM flags WHERE message_id = ?1 AND flag = ?2",
                rusqlite::params![message_db_id, flag],
            )?;
        }
        Ok(())
    }

    // ── Expunge ─────────────────────────────────────────────────────

    pub fn expunge(&self, mailbox_id: i64) -> Result<Vec<u32>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT m.uid FROM messages m JOIN flags f ON m.id = f.message_id WHERE m.mailbox_id = ?1 AND f.flag = '\\Deleted' ORDER BY m.uid",
        )?;
        let uids: Vec<u32> = stmt
            .query_map([mailbox_id], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;
        if !uids.is_empty() {
            conn.execute(
                "DELETE FROM messages WHERE mailbox_id = ?1 AND id IN (SELECT m.id FROM messages m JOIN flags f ON m.id = f.message_id WHERE m.mailbox_id = ?1 AND f.flag = '\\Deleted')",
                [mailbox_id],
            )?;
        }
        Ok(uids)
    }

    // ── Copy ────────────────────────────────────────────────────────

    pub fn copy_messages(
        &self,
        src_mailbox_id: i64,
        uids: &[u32],
        dst_mailbox_name: &str,
    ) -> Result<Vec<(u32, u32)>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;

        let (dst_id, mut dst_uid_next): (i64, u32) = tx
            .query_row(
                "SELECT id, uid_next FROM mailboxes WHERE name = ?1",
                [dst_mailbox_name],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|_| StoreError::MailboxNotFound(dst_mailbox_name.to_string()))?;

        let mut mapping = Vec::with_capacity(uids.len());
        for &src_uid in uids {
            let msg: MessageRow = tx.query_row(
                "SELECT id, mailbox_id, uid, harmony_msg_id, internal_date, rfc822_size FROM messages WHERE mailbox_id = ?1 AND uid = ?2",
                rusqlite::params![src_mailbox_id, src_uid],
                row_to_message,
            ).map_err(|_| StoreError::MessageNotFound)?;

            let new_uid = dst_uid_next;
            dst_uid_next += 1;
            tx.execute(
                "INSERT INTO messages (mailbox_id, uid, harmony_msg_id, internal_date, rfc822_size) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![dst_id, new_uid, msg.harmony_msg_id.as_slice(), msg.internal_date, msg.rfc822_size],
            )?;
            let new_msg_id = tx.last_insert_rowid();
            tx.execute("INSERT INTO flags (message_id, flag) SELECT ?1, flag FROM flags WHERE message_id = ?2", rusqlite::params![new_msg_id, msg.id])?;
            mapping.push((src_uid, new_uid));
        }

        tx.execute(
            "UPDATE mailboxes SET uid_next = ?1 WHERE id = ?2",
            rusqlite::params![dst_uid_next, dst_id],
        )?;
        tx.commit()?;
        Ok(mapping)
    }

    // ── User operations ─────────────────────────────────────────────

    pub fn create_user(
        &self,
        username: &str,
        password: &str,
        harmony_address: &[u8; ADDRESS_HASH_LEN],
    ) -> Result<(), StoreError> {
        let hash = hash_password(password)?;
        let now = now_secs();
        let conn = self.conn.lock().unwrap();
        match conn.execute(
            "INSERT INTO users (username, password_hash, harmony_address, created_at) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![username, hash, harmony_address.as_slice(), now],
        ) {
            Ok(_) => Ok(()),
            Err(rusqlite::Error::SqliteFailure(err, _))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Err(StoreError::UserExists(username.to_string()))
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn authenticate(&self, username: &str, password: &str) -> Result<UserRow, StoreError> {
        let conn = self.conn.lock().unwrap();
        let (id, hash, harmony_address_blob, created_at): (i64, String, Vec<u8>, u64) = conn
            .query_row(
                "SELECT id, password_hash, harmony_address, created_at FROM users WHERE username = ?1",
                [username],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .map_err(|_| StoreError::AuthFailed)?;
        // Drop the lock before password verification (CPU-intensive)
        drop(conn);

        verify_password(password, &hash)?;

        let mut harmony_address = [0u8; ADDRESS_HASH_LEN];
        if harmony_address_blob.len() == ADDRESS_HASH_LEN {
            harmony_address.copy_from_slice(&harmony_address_blob);
        }
        Ok(UserRow {
            id,
            username: username.to_string(),
            harmony_address,
            created_at,
        })
    }

    pub fn get_user(&self, username: &str) -> Result<Option<UserRow>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, username, harmony_address, created_at FROM users WHERE username = ?1",
        )?;
        let row = stmt
            .query_row([username], |row| {
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
            })
            .optional()?;
        Ok(row)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn row_to_message(row: &rusqlite::Row) -> rusqlite::Result<MessageRow> {
    let harmony_blob: Vec<u8> = row.get(3)?;
    let mut harmony_msg_id = [0u8; MESSAGE_ID_LEN];
    if harmony_blob.len() == MESSAGE_ID_LEN {
        harmony_msg_id.copy_from_slice(&harmony_blob);
    }
    Ok(MessageRow {
        id: row.get(0)?,
        mailbox_id: row.get(1)?,
        uid: row.get(2)?,
        harmony_msg_id,
        internal_date: row.get(4)?,
        rfc822_size: row.get(5)?,
    })
}

fn imap_pattern_to_sql(pattern: &str) -> String {
    pattern
        .replace('%', "*PERCENT*")
        .replace('_', "\\_")
        .replace('*', "%")
        .replace("*PERCENT*", "%")
}

fn hash_password(password: &str) -> Result<String, StoreError> {
    use argon2::password_hash::{rand_core::OsRng, PasswordHasher, SaltString};
    use argon2::Argon2;
    let salt = SaltString::generate(&mut OsRng);
    let hash = Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| StoreError::HashError(e.to_string()))?;
    Ok(hash.to_string())
}

fn verify_password(password: &str, hash: &str) -> Result<(), StoreError> {
    use argon2::password_hash::{PasswordHash, PasswordVerifier};
    use argon2::Argon2;
    let parsed_hash = PasswordHash::new(hash).map_err(|e| StoreError::HashError(e.to_string()))?;
    Argon2::default()
        .verify_password(password.as_bytes(), &parsed_hash)
        .map_err(|_| StoreError::AuthFailed)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

use rusqlite::OptionalExtension;

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> ImapStore {
        let store = ImapStore::open_memory().unwrap();
        store.initialize_default_mailboxes().unwrap();
        store
    }

    #[test]
    fn default_mailboxes_created() {
        let store = test_store();
        for name in DEFAULT_MAILBOXES {
            assert!(
                store.get_mailbox(name).unwrap().is_some(),
                "missing: {name}"
            );
        }
    }

    #[test]
    fn create_and_delete_mailbox() {
        let store = test_store();
        store.create_mailbox("Archive").unwrap();
        assert!(store.get_mailbox("Archive").unwrap().is_some());
        store.delete_mailbox("Archive").unwrap();
        assert!(store.get_mailbox("Archive").unwrap().is_none());
    }

    #[test]
    fn cannot_delete_inbox() {
        let store = test_store();
        assert!(matches!(
            store.delete_mailbox("INBOX").unwrap_err(),
            StoreError::MailboxNotFound(_)
        ));
    }

    #[test]
    fn duplicate_mailbox_rejected() {
        let store = test_store();
        assert!(matches!(
            store.create_mailbox("INBOX").unwrap_err(),
            StoreError::MailboxExists(_)
        ));
    }

    #[test]
    fn list_mailboxes_pattern() {
        let store = test_store();
        assert_eq!(
            store.list_mailboxes("*").unwrap().len(),
            DEFAULT_MAILBOXES.len()
        );
        let inbox = store.list_mailboxes("INBOX").unwrap();
        assert_eq!(inbox.len(), 1);
        assert_eq!(inbox[0].name, "INBOX");
    }

    #[test]
    fn subscribe_unsubscribe() {
        let store = test_store();
        assert!(!store.get_mailbox("INBOX").unwrap().unwrap().subscribed);
        store.subscribe("INBOX").unwrap();
        assert!(store.get_mailbox("INBOX").unwrap().unwrap().subscribed);
        store.unsubscribe("INBOX").unwrap();
        assert!(!store.get_mailbox("INBOX").unwrap().unwrap().subscribed);
    }

    #[test]
    fn insert_message_assigns_sequential_uids() {
        let store = test_store();
        assert_eq!(
            store
                .insert_message("INBOX", &[0xAA; MESSAGE_ID_LEN], 1000, 500)
                .unwrap(),
            1
        );
        assert_eq!(
            store
                .insert_message("INBOX", &[0xBB; MESSAGE_ID_LEN], 1001, 600)
                .unwrap(),
            2
        );
        assert_eq!(
            store
                .insert_message("INBOX", &[0xCC; MESSAGE_ID_LEN], 1002, 700)
                .unwrap(),
            3
        );
        assert_eq!(store.get_mailbox("INBOX").unwrap().unwrap().uid_next, 4);
    }

    #[test]
    fn get_messages_ordered_by_uid() {
        let store = test_store();
        let mbox = store.get_mailbox("INBOX").unwrap().unwrap();
        store
            .insert_message("INBOX", &[1; MESSAGE_ID_LEN], 100, 10)
            .unwrap();
        store
            .insert_message("INBOX", &[2; MESSAGE_ID_LEN], 200, 20)
            .unwrap();
        let messages = store.get_messages(mbox.id).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].uid, 1);
        assert_eq!(messages[1].uid, 2);
    }

    #[test]
    fn get_message_by_uid() {
        let store = test_store();
        let mbox = store.get_mailbox("INBOX").unwrap().unwrap();
        store
            .insert_message("INBOX", &[0xAA; MESSAGE_ID_LEN], 100, 10)
            .unwrap();
        assert!(store.get_message_by_uid(mbox.id, 1).unwrap().is_some());
        assert!(store.get_message_by_uid(mbox.id, 99).unwrap().is_none());
    }

    #[test]
    fn get_messages_by_uid_range() {
        let store = test_store();
        let mbox = store.get_mailbox("INBOX").unwrap().unwrap();
        for i in 0..5 {
            store
                .insert_message("INBOX", &[i; MESSAGE_ID_LEN], 100 + i as u64, 10)
                .unwrap();
        }
        let range = store.get_messages_by_uid_range(mbox.id, 2, 4).unwrap();
        assert_eq!(range.len(), 3);
        assert_eq!(range[0].uid, 2);
        assert_eq!(range[2].uid, 4);
    }

    #[test]
    fn count_messages() {
        let store = test_store();
        let mbox = store.get_mailbox("INBOX").unwrap().unwrap();
        assert_eq!(store.count_messages(mbox.id).unwrap(), 0);
        store
            .insert_message("INBOX", &[1; MESSAGE_ID_LEN], 100, 10)
            .unwrap();
        store
            .insert_message("INBOX", &[2; MESSAGE_ID_LEN], 200, 20)
            .unwrap();
        assert_eq!(store.count_messages(mbox.id).unwrap(), 2);
    }

    #[test]
    fn flag_operations() {
        let store = test_store();
        store
            .insert_message("INBOX", &[1; MESSAGE_ID_LEN], 100, 10)
            .unwrap();
        let mbox = store.get_mailbox("INBOX").unwrap().unwrap();
        let msg = store
            .get_messages(mbox.id)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        assert!(store.get_flags(msg.id).unwrap().is_empty());
        store.add_flags(msg.id, &["\\Seen", "\\Flagged"]).unwrap();
        assert_eq!(store.get_flags(msg.id).unwrap().len(), 2);
        store.add_flags(msg.id, &["\\Seen"]).unwrap(); // idempotent
        assert_eq!(store.get_flags(msg.id).unwrap().len(), 2);
        store.remove_flags(msg.id, &["\\Seen"]).unwrap();
        assert_eq!(store.get_flags(msg.id).unwrap(), vec!["\\Flagged"]);
        store.set_flags(msg.id, &["\\Answered", "\\Draft"]).unwrap();
        let flags = store.get_flags(msg.id).unwrap();
        assert_eq!(flags.len(), 2);
        assert!(flags.contains(&"\\Answered".to_string()));
    }

    #[test]
    fn expunge_deletes_flagged_messages() {
        let store = test_store();
        let mbox = store.get_mailbox("INBOX").unwrap().unwrap();
        store
            .insert_message("INBOX", &[1; MESSAGE_ID_LEN], 100, 10)
            .unwrap();
        store
            .insert_message("INBOX", &[2; MESSAGE_ID_LEN], 200, 20)
            .unwrap();
        store
            .insert_message("INBOX", &[3; MESSAGE_ID_LEN], 300, 30)
            .unwrap();
        let messages = store.get_messages(mbox.id).unwrap();
        store.add_flags(messages[1].id, &["\\Deleted"]).unwrap();
        let expunged = store.expunge(mbox.id).unwrap();
        assert_eq!(expunged, vec![2]);
        assert_eq!(store.get_messages(mbox.id).unwrap().len(), 2);
    }

    #[test]
    fn copy_messages_between_mailboxes() {
        let store = test_store();
        store
            .insert_message("INBOX", &[1; MESSAGE_ID_LEN], 100, 10)
            .unwrap();
        store
            .insert_message("INBOX", &[2; MESSAGE_ID_LEN], 200, 20)
            .unwrap();
        let inbox = store.get_mailbox("INBOX").unwrap().unwrap();
        let msgs = store.get_messages(inbox.id).unwrap();
        store.add_flags(msgs[0].id, &["\\Seen"]).unwrap();
        let mapping = store.copy_messages(inbox.id, &[1, 2], "Sent").unwrap();
        assert_eq!(mapping, vec![(1, 1), (2, 2)]);
        let sent = store.get_mailbox("Sent").unwrap().unwrap();
        assert_eq!(store.get_messages(sent.id).unwrap().len(), 2);
        let copied_flags = store
            .get_flags(store.get_messages(sent.id).unwrap()[0].id)
            .unwrap();
        assert_eq!(copied_flags, vec!["\\Seen"]);
    }

    #[test]
    fn count_recent_and_unseen() {
        let store = test_store();
        let mbox = store.get_mailbox("INBOX").unwrap().unwrap();
        store
            .insert_message("INBOX", &[1; MESSAGE_ID_LEN], 100, 10)
            .unwrap();
        store
            .insert_message("INBOX", &[2; MESSAGE_ID_LEN], 200, 20)
            .unwrap();
        store
            .insert_message("INBOX", &[3; MESSAGE_ID_LEN], 300, 30)
            .unwrap();
        let msgs = store.get_messages(mbox.id).unwrap();
        store.add_flags(msgs[0].id, &["\\Recent"]).unwrap();
        store.add_flags(msgs[1].id, &["\\Seen"]).unwrap();
        assert_eq!(store.count_recent(mbox.id).unwrap(), 1);
        assert_eq!(store.count_unseen(mbox.id).unwrap(), 2);
    }

    #[test]
    fn user_create_and_authenticate() {
        let store = test_store();
        let addr = [0xAA; ADDRESS_HASH_LEN];
        store.create_user("alice", "s3cret", &addr).unwrap();
        let user = store.authenticate("alice", "s3cret").unwrap();
        assert_eq!(user.username, "alice");
        assert_eq!(user.harmony_address, addr);
        assert!(matches!(
            store.authenticate("alice", "wrong").unwrap_err(),
            StoreError::AuthFailed
        ));
        assert!(matches!(
            store.authenticate("bob", "anything").unwrap_err(),
            StoreError::AuthFailed
        ));
    }

    #[test]
    fn duplicate_user_rejected() {
        let store = test_store();
        store
            .create_user("alice", "pass", &[0xAA; ADDRESS_HASH_LEN])
            .unwrap();
        assert!(matches!(
            store
                .create_user("alice", "other", &[0xAA; ADDRESS_HASH_LEN])
                .unwrap_err(),
            StoreError::UserExists(_)
        ));
    }

    #[test]
    fn get_user() {
        let store = test_store();
        assert!(store.get_user("nobody").unwrap().is_none());
        store
            .create_user("bob", "pass", &[0xBB; ADDRESS_HASH_LEN])
            .unwrap();
        let user = store.get_user("bob").unwrap().unwrap();
        assert_eq!(user.username, "bob");
    }

    #[test]
    fn initialize_is_idempotent() {
        let store = ImapStore::open_memory().unwrap();
        store.initialize_default_mailboxes().unwrap();
        store.initialize_default_mailboxes().unwrap();
        assert_eq!(
            store.list_mailboxes("*").unwrap().len(),
            DEFAULT_MAILBOXES.len()
        );
    }
}

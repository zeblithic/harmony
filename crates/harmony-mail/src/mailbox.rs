//! Merkle mailbox — CAS-backed email storage with folder structure.
//!
//! The mailbox is a tree of content-addressed objects:
//!
//! ```text
//! MailRoot (one per user, root CID published to Zenoh)
//!   ├─ inbox  → MailFolder CID
//!   ├─ sent   → MailFolder CID
//!   ├─ drafts → MailFolder CID
//!   └─ trash  → MailFolder CID
//!
//! MailFolder
//!   ├─ summary (count, unread)
//!   └─ pages: [MailPage CID, ...]
//!
//! MailPage (up to PAGE_CAPACITY message entries)
//!   ├─ entries: [MessageEntry, ...]
//!   └─ next: Option<MailPage CID>
//! ```
//!
//! All objects are serialized to bytes, stored via harmony-content, and
//! referenced by their 32-byte CID. The root CID is the only mutable
//! pointer; everything else is immutable.

use crate::error::MailError;
use crate::message::{ADDRESS_HASH_LEN, CID_LEN, MESSAGE_ID_LEN};

// ── Constants ──────────────────────────────────────────────────────────

/// Current mailbox format version.
pub const MAILBOX_VERSION: u8 = 0x01;

/// Magic bytes identifying a MailRoot blob.
pub const ROOT_MAGIC: [u8; 4] = *b"MBOX";

/// Magic bytes identifying a MailFolder blob.
pub const FOLDER_MAGIC: [u8; 4] = *b"MFLD";

/// Magic bytes identifying a MailPage blob.
pub const PAGE_MAGIC: [u8; 4] = *b"MPAG";

/// Maximum messages per page. Balances update cost (rehash one page on
/// new mail) against read cost (fetch one page to show a screenful).
/// ~100 entries * ~125 bytes each ≈ 12.5 KB per page blob.
pub const PAGE_CAPACITY: usize = 100;

/// Maximum subject snippet length stored in a MessageEntry.
/// Full subject lives in the HarmonyMessage blob itself.
pub const MAX_SNIPPET_LEN: usize = 128;

/// Number of standard folders.
pub const FOLDER_COUNT: usize = 4;

/// Folder names in canonical order (index matches `FolderKind` discriminant).
pub const FOLDER_NAMES: [&str; FOLDER_COUNT] = ["inbox", "sent", "drafts", "trash"];

/// Empty CID — all zeros, used as sentinel for "not yet created" folders/pages.
pub const EMPTY_CID: [u8; CID_LEN] = [0u8; CID_LEN];

// ── Folder kind ────────────────────────────────────────────────────────

/// Standard mailbox folders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FolderKind {
    Inbox = 0,
    Sent = 1,
    Drafts = 2,
    Trash = 3,
}

impl FolderKind {
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::Inbox),
            1 => Some(Self::Sent),
            2 => Some(Self::Drafts),
            3 => Some(Self::Trash),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        FOLDER_NAMES[self as usize]
    }
}

// ── MailRoot ───────────────────────────────────────────────────────────

/// Root node of a user's mailbox. Published to Zenoh as the mutable pointer.
///
/// Wire format:
/// ```text
/// [4] magic ("MBOX")
/// [1] version
/// [16] owner_address
/// [8] updated_at (unix seconds, big-endian)
/// [32 * FOLDER_COUNT] folder CIDs (inbox, sent, drafts, trash)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MailRoot {
    pub version: u8,
    pub owner_address: [u8; ADDRESS_HASH_LEN],
    pub updated_at: u64,
    pub folders: [[u8; CID_LEN]; FOLDER_COUNT],
}

impl MailRoot {
    /// Size of a serialized MailRoot in bytes.
    pub const WIRE_SIZE: usize = 4 + 1 + ADDRESS_HASH_LEN + 8 + (CID_LEN * FOLDER_COUNT);

    /// Create an empty mailbox for a given owner.
    pub fn new_empty(owner_address: [u8; ADDRESS_HASH_LEN], now: u64) -> Self {
        Self {
            version: MAILBOX_VERSION,
            owner_address,
            updated_at: now,
            folders: [EMPTY_CID; FOLDER_COUNT],
        }
    }

    /// Get the CID for a specific folder.
    pub fn folder_cid(&self, kind: FolderKind) -> &[u8; CID_LEN] {
        &self.folders[kind as usize]
    }

    /// Set the CID for a specific folder, returning the updated root.
    pub fn with_folder(mut self, kind: FolderKind, cid: [u8; CID_LEN], now: u64) -> Self {
        self.folders[kind as usize] = cid;
        self.updated_at = now;
        self
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::WIRE_SIZE);
        buf.extend_from_slice(&ROOT_MAGIC);
        buf.push(self.version);
        buf.extend_from_slice(&self.owner_address);
        buf.extend_from_slice(&self.updated_at.to_be_bytes());
        for folder_cid in &self.folders {
            buf.extend_from_slice(folder_cid);
        }
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, MailError> {
        if data.len() < Self::WIRE_SIZE {
            return Err(MailError::MessageTooShort {
                len: data.len(),
                min: Self::WIRE_SIZE,
            });
        }
        if &data[0..4] != &ROOT_MAGIC {
            return Err(MailError::UnsupportedVersion(data[0]));
        }
        let version = data[4];
        if version != MAILBOX_VERSION {
            return Err(MailError::UnsupportedVersion(version));
        }

        let mut pos = 5;
        let mut owner_address = [0u8; ADDRESS_HASH_LEN];
        owner_address.copy_from_slice(&data[pos..pos + ADDRESS_HASH_LEN]);
        pos += ADDRESS_HASH_LEN;

        let updated_at = u64::from_be_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let mut folders = [EMPTY_CID; FOLDER_COUNT];
        for folder in &mut folders {
            folder.copy_from_slice(&data[pos..pos + CID_LEN]);
            pos += CID_LEN;
        }

        Ok(Self {
            version,
            owner_address,
            updated_at,
            folders,
        })
    }
}

// ── MailFolder ─────────────────────────────────────────────────────────

/// A folder containing summary counts and references to pages of messages.
///
/// Wire format:
/// ```text
/// [4] magic ("MFLD")
/// [1] version
/// [4] message_count (big-endian u32)
/// [4] unread_count (big-endian u32)
/// [2] page_count (big-endian u16)
/// [32 * N] page CIDs (newest first — page 0 is the most recent)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MailFolder {
    pub version: u8,
    pub message_count: u32,
    pub unread_count: u32,
    pub page_cids: Vec<[u8; CID_LEN]>,
}

impl MailFolder {
    /// Minimum size: magic + version + counts + page_count.
    const MIN_SIZE: usize = 4 + 1 + 4 + 4 + 2;

    /// Create an empty folder.
    pub fn new_empty() -> Self {
        Self {
            version: MAILBOX_VERSION,
            message_count: 0,
            unread_count: 0,
            page_cids: Vec::new(),
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::MIN_SIZE + self.page_cids.len() * CID_LEN);
        buf.extend_from_slice(&FOLDER_MAGIC);
        buf.push(self.version);
        buf.extend_from_slice(&self.message_count.to_be_bytes());
        buf.extend_from_slice(&self.unread_count.to_be_bytes());
        buf.extend_from_slice(&(self.page_cids.len() as u16).to_be_bytes());
        for cid in &self.page_cids {
            buf.extend_from_slice(cid);
        }
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, MailError> {
        if data.len() < Self::MIN_SIZE {
            return Err(MailError::MessageTooShort {
                len: data.len(),
                min: Self::MIN_SIZE,
            });
        }
        if &data[0..4] != &FOLDER_MAGIC {
            return Err(MailError::UnsupportedVersion(data[0]));
        }
        let version = data[4];
        if version != MAILBOX_VERSION {
            return Err(MailError::UnsupportedVersion(version));
        }

        let mut pos = 5;
        let message_count = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let unread_count = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let page_count = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        let expected = pos + page_count * CID_LEN;
        if data.len() < expected {
            return Err(MailError::Truncated {
                expected: expected - data.len(),
            });
        }

        let mut page_cids = Vec::with_capacity(page_count);
        for _ in 0..page_count {
            let mut cid = [0u8; CID_LEN];
            cid.copy_from_slice(&data[pos..pos + CID_LEN]);
            pos += CID_LEN;
            page_cids.push(cid);
        }

        Ok(Self {
            version,
            message_count,
            unread_count,
            page_cids,
        })
    }
}

// ── MessageEntry ───────────────────────────────────────────────────────

/// Lightweight reference to a stored message — carried inside a MailPage.
/// Contains just enough metadata for inbox listing without fetching the
/// full HarmonyMessage blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageEntry {
    /// CID of the serialized HarmonyMessage blob in CAS.
    pub message_cid: [u8; CID_LEN],
    /// The message's unique ID (for threading and deduplication).
    pub message_id: [u8; MESSAGE_ID_LEN],
    /// Sender's address hash.
    pub sender_address: [u8; ADDRESS_HASH_LEN],
    /// Unix timestamp (seconds).
    pub timestamp: u64,
    /// Subject snippet (truncated to MAX_SNIPPET_LEN).
    pub subject_snippet: String,
    /// Read/unread state.
    pub read: bool,
}

impl MessageEntry {
    /// Minimum wire size: cid + message_id + sender + timestamp + flags + snippet_len.
    const MIN_SIZE: usize = CID_LEN + MESSAGE_ID_LEN + ADDRESS_HASH_LEN + 8 + 1 + 2;

    pub fn to_bytes(&self) -> Vec<u8> {
        let snippet_bytes = self.subject_snippet.as_bytes();
        let snippet_len = snippet_bytes.len().min(MAX_SNIPPET_LEN);
        let mut buf = Vec::with_capacity(Self::MIN_SIZE + snippet_len);

        buf.extend_from_slice(&self.message_cid);
        buf.extend_from_slice(&self.message_id);
        buf.extend_from_slice(&self.sender_address);
        buf.extend_from_slice(&self.timestamp.to_be_bytes());

        let flags: u8 = if self.read { 0x01 } else { 0x00 };
        buf.push(flags);

        buf.extend_from_slice(&(snippet_len as u16).to_be_bytes());
        buf.extend_from_slice(&snippet_bytes[..snippet_len]);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<(Self, usize), MailError> {
        if data.len() < Self::MIN_SIZE {
            return Err(MailError::MessageTooShort {
                len: data.len(),
                min: Self::MIN_SIZE,
            });
        }

        let mut pos = 0;

        let mut message_cid = [0u8; CID_LEN];
        message_cid.copy_from_slice(&data[pos..pos + CID_LEN]);
        pos += CID_LEN;

        let mut message_id = [0u8; MESSAGE_ID_LEN];
        message_id.copy_from_slice(&data[pos..pos + MESSAGE_ID_LEN]);
        pos += MESSAGE_ID_LEN;

        let mut sender_address = [0u8; ADDRESS_HASH_LEN];
        sender_address.copy_from_slice(&data[pos..pos + ADDRESS_HASH_LEN]);
        pos += ADDRESS_HASH_LEN;

        let timestamp = u64::from_be_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let read = data[pos] & 0x01 != 0;
        pos += 1;

        let snippet_len = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        if data.len() < pos + snippet_len {
            return Err(MailError::Truncated {
                expected: snippet_len,
            });
        }

        let subject_snippet = core::str::from_utf8(&data[pos..pos + snippet_len])
            .map_err(|_| MailError::InvalidUtf8 {
                field: "subject_snippet",
            })?
            .to_string();
        pos += snippet_len;

        Ok((
            Self {
                message_cid,
                message_id,
                sender_address,
                timestamp,
                subject_snippet,
                read,
            },
            pos,
        ))
    }
}

// ── MailPage ───────────────────────────────────────────────────────────

/// A page of message entries within a folder. Pages are linked — the
/// newest page is referenced from the folder, and each page links to
/// the next (older) page.
///
/// Wire format:
/// ```text
/// [4] magic ("MPAG")
/// [1] version
/// [1] has_next (0x00 | 0x01)
/// [32?] next_page_cid (if has_next)
/// [2] entry_count (big-endian u16)
/// [variable * N] entries (MessageEntry wire format)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MailPage {
    pub version: u8,
    pub next_page: Option<[u8; CID_LEN]>,
    pub entries: Vec<MessageEntry>,
}

impl MailPage {
    const MIN_SIZE: usize = 4 + 1 + 1 + 2; // magic + version + has_next + entry_count

    /// Create an empty page with no next link.
    pub fn new_empty() -> Self {
        Self {
            version: MAILBOX_VERSION,
            next_page: None,
            entries: Vec::new(),
        }
    }

    /// Whether this page is full and a new one should be created.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= PAGE_CAPACITY
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(&PAGE_MAGIC);
        buf.push(self.version);

        match &self.next_page {
            Some(cid) => {
                buf.push(0x01);
                buf.extend_from_slice(cid);
            }
            None => buf.push(0x00),
        }

        buf.extend_from_slice(&(self.entries.len() as u16).to_be_bytes());
        for entry in &self.entries {
            buf.extend_from_slice(&entry.to_bytes());
        }
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, MailError> {
        if data.len() < Self::MIN_SIZE {
            return Err(MailError::MessageTooShort {
                len: data.len(),
                min: Self::MIN_SIZE,
            });
        }
        if &data[0..4] != &PAGE_MAGIC {
            return Err(MailError::UnsupportedVersion(data[0]));
        }
        let version = data[4];
        if version != MAILBOX_VERSION {
            return Err(MailError::UnsupportedVersion(version));
        }

        let mut pos = 5;
        let has_next = data[pos];
        pos += 1;

        let next_page = if has_next == 0x01 {
            if data.len() < pos + CID_LEN {
                return Err(MailError::Truncated { expected: CID_LEN });
            }
            let mut cid = [0u8; CID_LEN];
            cid.copy_from_slice(&data[pos..pos + CID_LEN]);
            pos += CID_LEN;
            Some(cid)
        } else {
            None
        };

        if data.len() < pos + 2 {
            return Err(MailError::Truncated { expected: 2 });
        }
        let entry_count = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let (entry, consumed) = MessageEntry::from_bytes(&data[pos..])?;
            pos += consumed;
            entries.push(entry);
        }

        Ok(Self {
            version,
            next_page,
            entries,
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_address() -> [u8; ADDRESS_HASH_LEN] {
        let mut addr = [0u8; ADDRESS_HASH_LEN];
        addr[0] = 0xAB;
        addr[15] = 0xCD;
        addr
    }

    fn dummy_cid(tag: u8) -> [u8; CID_LEN] {
        let mut cid = [0u8; CID_LEN];
        cid[0] = tag;
        cid[31] = tag;
        cid
    }

    fn dummy_entry(tag: u8) -> MessageEntry {
        MessageEntry {
            message_cid: dummy_cid(tag),
            message_id: [tag; MESSAGE_ID_LEN],
            sender_address: dummy_address(),
            timestamp: 1744403200 + tag as u64,
            subject_snippet: format!("Test email #{tag}"),
            read: tag % 2 == 0,
        }
    }

    #[test]
    fn mail_root_roundtrip() {
        let root = MailRoot {
            version: MAILBOX_VERSION,
            owner_address: dummy_address(),
            updated_at: 1744403200,
            folders: [dummy_cid(1), dummy_cid(2), dummy_cid(3), dummy_cid(4)],
        };
        let bytes = root.to_bytes();
        assert_eq!(bytes.len(), MailRoot::WIRE_SIZE);
        let decoded = MailRoot::from_bytes(&bytes).unwrap();
        assert_eq!(root, decoded);
    }

    #[test]
    fn mail_root_empty() {
        let root = MailRoot::new_empty(dummy_address(), 1744403200);
        let bytes = root.to_bytes();
        let decoded = MailRoot::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.folders, [EMPTY_CID; FOLDER_COUNT]);
    }

    #[test]
    fn mail_root_with_folder() {
        let root = MailRoot::new_empty(dummy_address(), 100);
        let updated = root.with_folder(FolderKind::Inbox, dummy_cid(0xFF), 200);
        assert_eq!(updated.folders[0], dummy_cid(0xFF));
        assert_eq!(updated.updated_at, 200);
        // Other folders unchanged
        assert_eq!(updated.folders[1], EMPTY_CID);
    }

    #[test]
    fn mail_folder_roundtrip() {
        let folder = MailFolder {
            version: MAILBOX_VERSION,
            message_count: 42,
            unread_count: 3,
            page_cids: vec![dummy_cid(1), dummy_cid(2)],
        };
        let bytes = folder.to_bytes();
        let decoded = MailFolder::from_bytes(&bytes).unwrap();
        assert_eq!(folder, decoded);
    }

    #[test]
    fn mail_folder_empty() {
        let folder = MailFolder::new_empty();
        let bytes = folder.to_bytes();
        let decoded = MailFolder::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.message_count, 0);
        assert_eq!(decoded.unread_count, 0);
        assert!(decoded.page_cids.is_empty());
    }

    #[test]
    fn message_entry_roundtrip() {
        let entry = dummy_entry(7);
        let bytes = entry.to_bytes();
        let (decoded, consumed) = MessageEntry::from_bytes(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(entry, decoded);
    }

    #[test]
    fn mail_page_roundtrip() {
        let page = MailPage {
            version: MAILBOX_VERSION,
            next_page: Some(dummy_cid(0xAA)),
            entries: vec![dummy_entry(1), dummy_entry(2), dummy_entry(3)],
        };
        let bytes = page.to_bytes();
        let decoded = MailPage::from_bytes(&bytes).unwrap();
        assert_eq!(page, decoded);
    }

    #[test]
    fn mail_page_no_next() {
        let page = MailPage {
            version: MAILBOX_VERSION,
            next_page: None,
            entries: vec![dummy_entry(5)],
        };
        let bytes = page.to_bytes();
        let decoded = MailPage::from_bytes(&bytes).unwrap();
        assert!(decoded.next_page.is_none());
        assert_eq!(decoded.entries.len(), 1);
    }

    #[test]
    fn mail_page_is_full() {
        let mut page = MailPage::new_empty();
        assert!(!page.is_full());
        for i in 0..PAGE_CAPACITY {
            page.entries.push(dummy_entry(i as u8));
        }
        assert!(page.is_full());
    }

    #[test]
    fn folder_kind_roundtrip() {
        for i in 0..FOLDER_COUNT {
            let kind = FolderKind::from_u8(i as u8).unwrap();
            assert_eq!(kind as u8, i as u8);
            assert_eq!(kind.name(), FOLDER_NAMES[i]);
        }
        assert!(FolderKind::from_u8(4).is_none());
    }
}

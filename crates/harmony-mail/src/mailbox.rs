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

// ── Wire format v1 size guards ───────────────────────────────────────
// If any of these external constants change, the v1 wire format breaks.
// Fail at compile time rather than producing silently incompatible blobs.
const _: () = assert!(CID_LEN == 32, "mailbox v1 wire format requires CID_LEN == 32");
const _: () = assert!(MESSAGE_ID_LEN == 16, "mailbox v1 wire format requires MESSAGE_ID_LEN == 16");
const _: () = assert!(ADDRESS_HASH_LEN == 16, "mailbox v1 wire format requires ADDRESS_HASH_LEN == 16");

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

// ── Helpers ────────────────────────────────────────────────────────────

/// Truncate a UTF-8 string to at most `max_bytes` without splitting
/// multi-byte characters. Returns the longest valid prefix.
pub(crate) fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    // Walk backwards from the byte limit to find a char boundary.
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

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
        buf.push(MAILBOX_VERSION);
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
        if data.len() > Self::WIRE_SIZE {
            return Err(MailError::TrailingBytes {
                count: data.len() - Self::WIRE_SIZE,
            });
        }

        let mut found_magic = [0u8; 4];
        found_magic.copy_from_slice(&data[0..4]);
        if found_magic != ROOT_MAGIC {
            return Err(MailError::InvalidMagic {
                expected: ROOT_MAGIC,
                found: found_magic,
            });
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

    pub fn to_bytes(&self) -> Result<Vec<u8>, MailError> {
        if self.unread_count > self.message_count {
            return Err(MailError::TooManyEntries {
                count: self.unread_count as usize,
                max: self.message_count as usize,
            });
        }
        let count = u16::try_from(self.page_cids.len()).map_err(|_| {
            MailError::TooManyEntries {
                count: self.page_cids.len(),
                max: u16::MAX as usize,
            }
        })?;
        let mut buf = Vec::with_capacity(Self::MIN_SIZE + self.page_cids.len() * CID_LEN);
        buf.extend_from_slice(&FOLDER_MAGIC);
        buf.push(MAILBOX_VERSION);
        buf.extend_from_slice(&self.message_count.to_be_bytes());
        buf.extend_from_slice(&self.unread_count.to_be_bytes());
        buf.extend_from_slice(&count.to_be_bytes());
        for cid in &self.page_cids {
            buf.extend_from_slice(cid);
        }
        Ok(buf)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, MailError> {
        if data.len() < Self::MIN_SIZE {
            return Err(MailError::MessageTooShort {
                len: data.len(),
                min: Self::MIN_SIZE,
            });
        }

        let mut found_magic = [0u8; 4];
        found_magic.copy_from_slice(&data[0..4]);
        if found_magic != FOLDER_MAGIC {
            return Err(MailError::InvalidMagic {
                expected: FOLDER_MAGIC,
                found: found_magic,
            });
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

        if unread_count > message_count {
            return Err(MailError::TooManyEntries {
                count: unread_count as usize,
                max: message_count as usize,
            });
        }

        let page_count = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        let expected = pos + page_count * CID_LEN;
        if data.len() < expected {
            return Err(MailError::Truncated {
                expected: expected - data.len(),
            });
        }
        if data.len() > expected {
            return Err(MailError::TrailingBytes {
                count: data.len() - expected,
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
        let snippet = truncate_utf8(&self.subject_snippet, MAX_SNIPPET_LEN);
        let snippet_bytes = snippet.as_bytes();
        let mut buf = Vec::with_capacity(Self::MIN_SIZE + snippet_bytes.len());

        buf.extend_from_slice(&self.message_cid);
        buf.extend_from_slice(&self.message_id);
        buf.extend_from_slice(&self.sender_address);
        buf.extend_from_slice(&self.timestamp.to_be_bytes());

        let flags: u8 = if self.read { 0x01 } else { 0x00 };
        buf.push(flags);

        buf.extend_from_slice(&(snippet_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(snippet_bytes);
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

        let read_flag = data[pos];
        if read_flag != 0x00 && read_flag != 0x01 {
            return Err(MailError::InvalidFlag {
                field: "read",
                value: read_flag,
            });
        }
        let read = read_flag == 0x01;
        pos += 1;

        let snippet_len = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        if snippet_len > MAX_SNIPPET_LEN {
            return Err(MailError::FieldTooLong {
                field: "subject_snippet",
                len: snippet_len,
                max: MAX_SNIPPET_LEN,
            });
        }

        if data.len() < pos + snippet_len {
            return Err(MailError::Truncated {
                expected: (pos + snippet_len) - data.len(),
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

/// A page of message entries within a folder.
///
/// NOTE: The folder's `page_cids` array is the canonical page index.
/// The `next_page` field provides redundant linked-list traversal and
/// must stay in sync with `page_cids` ordering. See ZEB-101 for the
/// decision on whether to remove this dual linkage.
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

    pub fn to_bytes(&self) -> Result<Vec<u8>, MailError> {
        if self.entries.len() > PAGE_CAPACITY {
            return Err(MailError::TooManyEntries {
                count: self.entries.len(),
                max: PAGE_CAPACITY,
            });
        }
        let count = u16::try_from(self.entries.len()).map_err(|_| {
            MailError::TooManyEntries {
                count: self.entries.len(),
                max: u16::MAX as usize,
            }
        })?;

        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(&PAGE_MAGIC);
        buf.push(MAILBOX_VERSION);

        match &self.next_page {
            Some(cid) => {
                buf.push(0x01);
                buf.extend_from_slice(cid);
            }
            None => buf.push(0x00),
        }

        buf.extend_from_slice(&count.to_be_bytes());
        for entry in &self.entries {
            buf.extend_from_slice(&entry.to_bytes());
        }
        Ok(buf)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, MailError> {
        if data.len() < Self::MIN_SIZE {
            return Err(MailError::MessageTooShort {
                len: data.len(),
                min: Self::MIN_SIZE,
            });
        }

        let mut found_magic = [0u8; 4];
        found_magic.copy_from_slice(&data[0..4]);
        if found_magic != PAGE_MAGIC {
            return Err(MailError::InvalidMagic {
                expected: PAGE_MAGIC,
                found: found_magic,
            });
        }
        let version = data[4];
        if version != MAILBOX_VERSION {
            return Err(MailError::UnsupportedVersion(version));
        }

        let mut pos = 5;
        let has_next = data[pos];
        pos += 1;

        let next_page = match has_next {
            0x00 => None,
            0x01 => {
                if data.len() < pos + CID_LEN {
                    return Err(MailError::Truncated {
                        expected: (pos + CID_LEN) - data.len(),
                    });
                }
                let mut cid = [0u8; CID_LEN];
                cid.copy_from_slice(&data[pos..pos + CID_LEN]);
                pos += CID_LEN;
                Some(cid)
            }
            _ => {
                return Err(MailError::InvalidFlag {
                    field: "has_next",
                    value: has_next,
                });
            }
        };

        if data.len() < pos + 2 {
            return Err(MailError::Truncated {
                expected: (pos + 2) - data.len(),
            });
        }
        let entry_count = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        if entry_count > PAGE_CAPACITY {
            return Err(MailError::TooManyEntries {
                count: entry_count,
                max: PAGE_CAPACITY,
            });
        }

        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let (entry, consumed) = MessageEntry::from_bytes(&data[pos..])?;
            pos += consumed;
            entries.push(entry);
        }

        if pos != data.len() {
            return Err(MailError::TrailingBytes {
                count: data.len() - pos,
            });
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
    fn mail_root_rejects_trailing_bytes() {
        let root = MailRoot::new_empty(dummy_address(), 100);
        let mut bytes = root.to_bytes();
        bytes.push(0xFF);
        assert!(matches!(
            MailRoot::from_bytes(&bytes),
            Err(MailError::TrailingBytes { count: 1 })
        ));
    }

    #[test]
    fn mail_root_rejects_bad_magic() {
        let mut bytes = MailRoot::new_empty(dummy_address(), 100).to_bytes();
        bytes[0] = b'X';
        assert!(matches!(
            MailRoot::from_bytes(&bytes),
            Err(MailError::InvalidMagic { .. })
        ));
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
        let bytes = folder.to_bytes().unwrap();
        let decoded = MailFolder::from_bytes(&bytes).unwrap();
        assert_eq!(folder, decoded);
    }

    #[test]
    fn mail_folder_rejects_trailing_bytes() {
        let folder = MailFolder::new_empty();
        let mut bytes = folder.to_bytes().unwrap();
        bytes.push(0xFF);
        assert!(matches!(
            MailFolder::from_bytes(&bytes),
            Err(MailError::TrailingBytes { count: 1 })
        ));
    }

    #[test]
    fn mail_folder_empty() {
        let folder = MailFolder::new_empty();
        let bytes = folder.to_bytes().unwrap();
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
    fn message_entry_utf8_truncation() {
        // 4-byte UTF-8 char (emoji) right at the boundary
        let entry = MessageEntry {
            message_cid: dummy_cid(1),
            message_id: [1; MESSAGE_ID_LEN],
            sender_address: dummy_address(),
            timestamp: 100,
            // Fill to just over MAX_SNIPPET_LEN with multi-byte chars
            subject_snippet: "x".repeat(MAX_SNIPPET_LEN - 1) + "\u{1F600}", // 127 + 4 = 131 bytes
            read: false,
        };
        let bytes = entry.to_bytes();
        // Should roundtrip without InvalidUtf8 — truncation respects char boundary
        let (decoded, _) = MessageEntry::from_bytes(&bytes).unwrap();
        assert!(decoded.subject_snippet.len() <= MAX_SNIPPET_LEN);
        assert!(decoded.subject_snippet.is_char_boundary(decoded.subject_snippet.len()));
    }

    #[test]
    fn mail_page_roundtrip() {
        let page = MailPage {
            version: MAILBOX_VERSION,
            next_page: Some(dummy_cid(0xAA)),
            entries: vec![dummy_entry(1), dummy_entry(2), dummy_entry(3)],
        };
        let bytes = page.to_bytes().unwrap();
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
        let bytes = page.to_bytes().unwrap();
        let decoded = MailPage::from_bytes(&bytes).unwrap();
        assert!(decoded.next_page.is_none());
        assert_eq!(decoded.entries.len(), 1);
    }

    #[test]
    fn mail_page_rejects_invalid_has_next() {
        let page = MailPage::new_empty();
        let mut bytes = page.to_bytes().unwrap();
        // has_next byte is at offset 5 (after magic + version)
        bytes[5] = 0x02;
        assert!(matches!(
            MailPage::from_bytes(&bytes),
            Err(MailError::InvalidFlag { field: "has_next", value: 0x02 })
        ));
    }

    #[test]
    fn mail_page_rejects_trailing_bytes() {
        let page = MailPage {
            version: MAILBOX_VERSION,
            next_page: None,
            entries: vec![dummy_entry(1)],
        };
        let mut bytes = page.to_bytes().unwrap();
        bytes.push(0xFF);
        assert!(matches!(
            MailPage::from_bytes(&bytes),
            Err(MailError::TrailingBytes { .. })
        ));
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

    #[test]
    fn message_entry_rejects_invalid_read_flag() {
        let entry = dummy_entry(1);
        let mut bytes = entry.to_bytes();
        // read flag is at offset: CID + message_id + sender + timestamp
        let flag_offset = CID_LEN + MESSAGE_ID_LEN + ADDRESS_HASH_LEN + 8;
        bytes[flag_offset] = 0x02;
        assert!(matches!(
            MessageEntry::from_bytes(&bytes),
            Err(MailError::InvalidFlag { field: "read", value: 0x02 })
        ));
    }

    #[test]
    fn mail_page_rejects_entry_count_over_capacity() {
        let page = MailPage {
            version: MAILBOX_VERSION,
            next_page: None,
            entries: vec![dummy_entry(1)],
        };
        let mut bytes = page.to_bytes().unwrap();
        // entry_count is a u16 at offset 6 (magic:4 + version:1 + has_next:1)
        let count_offset = 6;
        let bad_count = (PAGE_CAPACITY as u16) + 1;
        bytes[count_offset..count_offset + 2].copy_from_slice(&bad_count.to_be_bytes());
        assert!(matches!(
            MailPage::from_bytes(&bytes),
            Err(MailError::TooManyEntries { .. })
        ));
    }

    #[test]
    fn message_entry_rejects_oversized_snippet() {
        let entry = dummy_entry(1);
        let mut bytes = entry.to_bytes();
        // snippet_len is a u16 at offset: CID + message_id + sender + timestamp + flags
        let len_offset = CID_LEN + MESSAGE_ID_LEN + ADDRESS_HASH_LEN + 8 + 1;
        let bad_len = (MAX_SNIPPET_LEN as u16) + 1;
        bytes[len_offset..len_offset + 2].copy_from_slice(&bad_len.to_be_bytes());
        assert!(matches!(
            MessageEntry::from_bytes(&bytes),
            Err(MailError::FieldTooLong { field: "subject_snippet", max: MAX_SNIPPET_LEN, .. })
        ));
    }

    #[test]
    fn mail_folder_rejects_unread_exceeding_total_on_deserialize() {
        // Manually craft bytes with unread > total to test from_bytes rejection
        let mut folder = MailFolder::new_empty();
        folder.message_count = 5;
        folder.unread_count = 3; // valid for to_bytes
        let mut bytes = folder.to_bytes().unwrap();
        // Patch unread_count (offset 9..13) to exceed message_count
        bytes[9..13].copy_from_slice(&10u32.to_be_bytes());
        assert!(matches!(
            MailFolder::from_bytes(&bytes),
            Err(MailError::TooManyEntries { .. })
        ));
    }

    #[test]
    fn mail_folder_to_bytes_rejects_unread_exceeding_total() {
        let mut folder = MailFolder::new_empty();
        folder.message_count = 5;
        folder.unread_count = 10; // invalid: unread > total
        assert!(matches!(
            folder.to_bytes(),
            Err(MailError::TooManyEntries { .. })
        ));
    }

    #[test]
    fn truncate_utf8_respects_boundaries() {
        assert_eq!(truncate_utf8("hello", 10), "hello");
        assert_eq!(truncate_utf8("hello", 3), "hel");
        // 2-byte char: é = 0xC3 0xA9
        assert_eq!(truncate_utf8("café", 3), "caf");
        assert_eq!(truncate_utf8("café", 4), "caf");
        assert_eq!(truncate_utf8("café", 5), "café");
        // 4-byte char: 😀 = F0 9F 98 80
        assert_eq!(truncate_utf8("a😀b", 2), "a");
        assert_eq!(truncate_utf8("a😀b", 5), "a😀");
    }
}

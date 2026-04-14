//! Shared mailbox wire format types for Harmony mail.
//!
//! This crate provides the wire-format types used by both the `harmony-mail`
//! gateway and `harmony-client`, allowing the client to decode mailbox and
//! message data without depending on the heavier `harmony-mail` crate.

pub mod error;
pub mod mailbox;
pub mod message;

pub use error::MailboxError;
pub use mailbox::{
    FolderKind, MailFolder, MailPage, MailRoot, MessageEntry, EMPTY_CID, FOLDER_COUNT,
    FOLDER_MAGIC, FOLDER_NAMES, MAILBOX_VERSION, MAX_SNIPPET_LEN, PAGE_CAPACITY, PAGE_MAGIC,
    ROOT_MAGIC,
};
pub use message::{
    AttachmentRef, HarmonyMessage, MailMessageType, MessageFlags, Recipient, ADDRESS_HASH_LEN,
    CID_LEN, MAX_ATTACHMENTS, MAX_BODY_LEN, MAX_RECIPIENTS, MAX_SUBJECT_LEN, MESSAGE_ID_LEN,
    VERSION,
};

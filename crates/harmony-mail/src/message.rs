//! Harmony-native email message format.
//!
//! Wire format types (including `unique_message_id`) live in the
//! `harmony-mailbox` crate. This module re-exports them so existing
//! `harmony_mail::message::*` consumers continue to compile.

pub use harmony_mailbox::message::*;

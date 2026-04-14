//! Harmony-native email message format.
//!
//! Wire format types are defined in the `harmony-mailbox` crate.
//! This module re-exports them and adds `unique_message_id()`.

pub use harmony_mailbox::message::*;

/// Generate a unique 16-byte message ID using timestamp + atomic counter.
/// Safe to call from multiple threads -- uses a process-global atomic counter
/// to guarantee uniqueness even within the same nanosecond.
pub fn unique_message_id() -> [u8; MESSAGE_ID_LEN] {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);

    let mut hasher = blake3::Hasher::new();
    hasher.update(&now.to_le_bytes());
    hasher.update(&seq.to_le_bytes());
    let hash = hasher.finalize();
    let mut id = [0u8; MESSAGE_ID_LEN];
    id.copy_from_slice(&hash.as_bytes()[..MESSAGE_ID_LEN]);
    id
}

//! TTL + LRU caches for the resolver (spec §5.2).
//!
//! Four positive caches (claim, signing-key, master-key, revocation),
//! one negative cache, and one "domain last seen" sliding-window cache
//! for the 72h soft-fail. Time is injected via `TimeSource` so tests
//! control expiry with `FakeTimeSource::advance`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Clock abstraction. Production uses `SystemTimeSource`; tests use
/// `FakeTimeSource` (see `test_support`).
pub trait TimeSource: Send + Sync + 'static {
    fn now(&self) -> u64;
}

pub struct SystemTimeSource;

impl TimeSource for SystemTimeSource {
    fn now(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

/// Cheap, monotonic counter used as an LRU-ish recency token. Not a
/// real LRU — we only need "evict oldest N when over bound," which a
/// monotonic counter + sort-and-drop achieves without a doubly-linked
/// list. Spec §7.8 allows coarse-grained eviction since the attack
/// model is "bound memory under adversarial input," not "serve hot
/// items preferentially."
#[derive(Debug)]
pub struct RecencyCounter(AtomicU64);

impl RecencyCounter {
    pub fn new() -> Self {
        Self(AtomicU64::new(1))
    }
    pub fn tick(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

impl Default for RecencyCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    pub value: T,
    pub expires_at: u64,
    pub recency: u64,
}

impl<T> CacheEntry<T> {
    pub fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::FakeTimeSource;

    #[test]
    fn cache_entry_is_expired_at_boundary() {
        let entry = CacheEntry {
            value: 42u32,
            expires_at: 100,
            recency: 1,
        };
        assert!(!entry.is_expired(99));
        assert!(entry.is_expired(100));
        assert!(entry.is_expired(101));
    }

    #[test]
    fn fake_time_source_advances() {
        let t = FakeTimeSource::new(1000);
        assert_eq!(t.now(), 1000);
        t.advance(500);
        assert_eq!(t.now(), 1500);
    }
}

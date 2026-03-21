//! Cache lifecycle state machine.
//!
//! A sans-I/O cache lifecycle manager. [`CacheManager`] tracks licensed content
//! cached on a consumer's node and emits [`CacheAction`] variants for the caller
//! to execute (file deletion, notifications, renewal requests).

use alloc::vec::Vec;
use harmony_content::ContentId;
use hashbrown::HashMap;

/// Lifecycle state of a cached content entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheState {
    /// License is valid; no action needed.
    Active,
    /// License is approaching expiry; consumer has been notified.
    Expiring,
}

/// A single cached content entry tracked by the [`CacheManager`].
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// CID of the license manifest.
    pub manifest_cid: ContentId,
    /// CID of the actual content blob.
    pub content_cid: ContentId,
    /// UCAN `not_after` timestamp. `None` means perpetual license.
    pub ucan_not_after: Option<f64>,
    /// How many seconds before expiry to send a notice.
    pub expiry_notice_secs: u32,
    /// Current lifecycle state.
    pub state: CacheState,
    /// Whether to automatically request renewal on expiry notice.
    pub auto_renew: bool,
}

/// Actions emitted by the cache state machine for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheAction {
    /// License is about to expire; notify the consumer.
    NotifyExpiring {
        /// Manifest CID of the expiring content.
        manifest_cid: ContentId,
        /// Seconds remaining until expiry.
        seconds_remaining: u32,
    },
    /// Auto-renew is enabled; request a license renewal.
    RequestRenewal {
        /// Manifest CID to renew.
        manifest_cid: ContentId,
    },
    /// Wipe the decryption key from memory/storage.
    WipeKey {
        /// Manifest CID whose key should be wiped.
        manifest_cid: ContentId,
    },
    /// Evict the cached content blob from storage.
    EvictContent {
        /// Content CID to evict.
        content_cid: ContentId,
    },
}

/// Sans-I/O cache lifecycle manager.
///
/// Tracks cached content entries keyed by manifest CID and emits
/// [`CacheAction`]s when driven by [`tick`](Self::tick).
///
/// ```text
/// Empty → (add_entry) → Active → (notice threshold) → Expiring → (expired) → Evict → removed
///                         ↑                              │
///                         └── (renewal success) ──────────┘
/// ```
pub struct CacheManager {
    entries: HashMap<ContentId, CacheEntry>,
}

impl CacheManager {
    /// Create an empty cache manager.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Insert a cache entry, keyed by its `manifest_cid`.
    pub fn add_entry(&mut self, entry: CacheEntry) {
        self.entries.insert(entry.manifest_cid, entry);
    }

    /// Number of entries currently tracked.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Handle a successful license renewal by resetting the entry to
    /// [`CacheState::Active`] with a new expiry deadline.
    pub fn handle_renewal_success(&mut self, manifest_cid: &ContentId, new_not_after: f64) {
        if let Some(entry) = self.entries.get_mut(manifest_cid) {
            entry.state = CacheState::Active;
            entry.ucan_not_after = Some(new_not_after);
        }
    }

    /// Immediately revoke a cached entry, returning actions to wipe key and
    /// evict content.
    pub fn revoke(&mut self, manifest_cid: &ContentId) -> Vec<CacheAction> {
        if let Some(entry) = self.entries.remove(manifest_cid) {
            alloc::vec![
                CacheAction::WipeKey {
                    manifest_cid: entry.manifest_cid,
                },
                CacheAction::EvictContent {
                    content_cid: entry.content_cid,
                },
            ]
        } else {
            Vec::new()
        }
    }

    /// Advance the state machine. Call periodically with the current timestamp.
    ///
    /// For each entry with a finite expiry (`ucan_not_after = Some(t)`):
    /// - **Active** entries transition to **Expiring** when the remaining time
    ///   drops to or below `expiry_notice_secs`, emitting [`CacheAction::NotifyExpiring`]
    ///   and optionally [`CacheAction::RequestRenewal`] if `auto_renew` is set.
    /// - **Expiring** entries that have passed their deadline emit
    ///   [`CacheAction::WipeKey`] + [`CacheAction::EvictContent`] and are removed.
    ///
    /// Perpetual entries (`ucan_not_after = None`) are never affected.
    pub fn tick(&mut self, now: f64) -> Vec<CacheAction> {
        let mut actions = Vec::new();
        let mut to_remove = Vec::new();

        for (manifest_cid, entry) in self.entries.iter_mut() {
            let not_after = match entry.ucan_not_after {
                Some(t) => t,
                None => continue, // perpetual — skip
            };

            match entry.state {
                CacheState::Active => {
                    let remaining = (not_after - now).max(0.0);
                    if remaining <= entry.expiry_notice_secs as f64 {
                        if now >= not_after {
                            // Already past deadline (e.g., node was offline) —
                            // skip the Expiring state and evict immediately.
                            // Still honor auto_renew so the user can re-acquire access.
                            if entry.auto_renew {
                                actions.push(CacheAction::RequestRenewal {
                                    manifest_cid: *manifest_cid,
                                });
                            }
                            actions.push(CacheAction::WipeKey {
                                manifest_cid: *manifest_cid,
                            });
                            actions.push(CacheAction::EvictContent {
                                content_cid: entry.content_cid,
                            });
                            to_remove.push(*manifest_cid);
                        } else {
                            entry.state = CacheState::Expiring;
                            actions.push(CacheAction::NotifyExpiring {
                                manifest_cid: *manifest_cid,
                                seconds_remaining: remaining as u32,
                            });
                            if entry.auto_renew {
                                actions.push(CacheAction::RequestRenewal {
                                    manifest_cid: *manifest_cid,
                                });
                            }
                        }
                    }
                }
                CacheState::Expiring => {
                    if now >= not_after {
                        actions.push(CacheAction::WipeKey {
                            manifest_cid: *manifest_cid,
                        });
                        actions.push(CacheAction::EvictContent {
                            content_cid: entry.content_cid,
                        });
                        to_remove.push(*manifest_cid);
                    }
                }
            }
        }

        for cid in to_remove {
            self.entries.remove(&cid);
        }

        actions
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::ContentFlags;

    fn make_cid(data: &[u8]) -> harmony_content::ContentId {
        harmony_content::ContentId::for_book(data, ContentFlags::default()).unwrap()
    }

    #[test]
    fn active_entry_transitions_to_expiring() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,

            ucan_not_after: Some(100.0),
            expiry_notice_secs: 10,
            state: CacheState::Active,
            auto_renew: false,
        });

        // At t=89, still 11s before expiry — no action.
        let actions = mgr.tick(89.0);
        assert!(actions.is_empty());

        // At t=91, only 9s left — should transition to Expiring.
        let actions = mgr.tick(91.0);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            CacheAction::NotifyExpiring { seconds_remaining, .. } if *seconds_remaining == 9
        ));
    }

    #[test]
    fn expiring_entry_evicts_on_deadline() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,

            ucan_not_after: Some(100.0),
            expiry_notice_secs: 10,
            state: CacheState::Active,
            auto_renew: false,
        });

        let _ = mgr.tick(91.0); // transitions to Expiring
        let actions = mgr.tick(101.0); // past not_after

        assert_eq!(actions.len(), 2);
        assert!(matches!(&actions[0], CacheAction::WipeKey { .. }));
        assert!(matches!(&actions[1], CacheAction::EvictContent { .. }));
        assert_eq!(mgr.entry_count(), 0);
    }

    #[test]
    fn auto_renew_requests_renewal() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,

            ucan_not_after: Some(100.0),
            expiry_notice_secs: 10,
            state: CacheState::Active,
            auto_renew: true,
        });

        let actions = mgr.tick(91.0);
        assert_eq!(actions.len(), 2);
        assert!(matches!(&actions[0], CacheAction::NotifyExpiring { .. }));
        assert!(matches!(&actions[1], CacheAction::RequestRenewal { .. }));
    }

    #[test]
    fn renewal_success_resets_to_active() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,

            ucan_not_after: Some(100.0),
            expiry_notice_secs: 10,
            state: CacheState::Active,
            auto_renew: true,
        });

        let _ = mgr.tick(91.0); // transitions to Expiring
        mgr.handle_renewal_success(&manifest_cid, 200.0);

        // Should be back to Active with new deadline.
        let actions = mgr.tick(95.0);
        assert!(actions.is_empty()); // Still well before new expiry
    }

    #[test]
    fn perpetual_license_never_expires() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,

            ucan_not_after: None, // perpetual
            expiry_notice_secs: 0,
            state: CacheState::Active,
            auto_renew: false,
        });

        let actions = mgr.tick(999_999_999.0);
        assert!(actions.is_empty());
        assert_eq!(mgr.entry_count(), 1);
    }

    #[test]
    fn active_entry_past_deadline_evicts_immediately() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,

            ucan_not_after: Some(100.0),
            expiry_notice_secs: 10,
            state: CacheState::Active,
            auto_renew: false,
        });

        // Node was offline and comes back at t=200, well past not_after=100.
        // Should evict in a single tick, not require two ticks.
        let actions = mgr.tick(200.0);
        assert_eq!(actions.len(), 2);
        assert!(matches!(&actions[0], CacheAction::WipeKey { .. }));
        assert!(matches!(&actions[1], CacheAction::EvictContent { .. }));
        assert_eq!(mgr.entry_count(), 0);
    }

    #[test]
    fn active_entry_past_deadline_with_auto_renew_requests_renewal() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,

            ucan_not_after: Some(100.0),
            expiry_notice_secs: 10,
            state: CacheState::Active,
            auto_renew: true,
        });

        // Node was offline, comes back past deadline. Should evict AND
        // request renewal since auto_renew is set.
        let actions = mgr.tick(200.0);
        assert_eq!(actions.len(), 3);
        assert!(matches!(&actions[0], CacheAction::RequestRenewal { .. }));
        assert!(matches!(&actions[1], CacheAction::WipeKey { .. }));
        assert!(matches!(&actions[2], CacheAction::EvictContent { .. }));
        assert_eq!(mgr.entry_count(), 0);
    }

    #[test]
    fn revoke_immediately_evicts() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,

            ucan_not_after: Some(100.0),
            expiry_notice_secs: 10,
            state: CacheState::Active,
            auto_renew: false,
        });

        let actions = mgr.revoke(&manifest_cid);
        assert_eq!(actions.len(), 2);
        assert!(matches!(&actions[0], CacheAction::WipeKey { .. }));
        assert!(matches!(&actions[1], CacheAction::EvictContent { .. }));
        assert_eq!(mgr.entry_count(), 0);
    }
}

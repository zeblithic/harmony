extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// A single vine in the feed.
#[derive(Debug, Clone, PartialEq)]
pub struct VineFeedItem {
    pub bundle_cid: [u8; 32],
    pub video_cid: [u8; 32],
    pub creator: [u8; 16],
    /// Unix timestamp in seconds when the vine was created.
    pub timestamp: u64,
    pub title: Option<String>,
    pub reshare_of: Option<[u8; 32]>,
}

/// Events the VineFeed state machine accepts.
#[derive(Debug, Clone)]
pub enum VineEvent {
    FollowCreator { address: [u8; 16] },
    UnfollowCreator { address: [u8; 16] },
    VineAnnounced { item: VineFeedItem },
    MarkViewed { bundle_cid: [u8; 32] },
    MarkAllViewed,
}

/// Actions the VineFeed state machine emits.
#[derive(Debug, Clone)]
pub enum VineAction {
    Subscribe { key_expr: String },
    Unsubscribe { key_expr: String },
    FeedUpdated,
}

/// Build the vine announce subscription pattern for a creator address.
fn vine_announce_pattern(addr_hex: &str) -> String {
    harmony_zenoh::keyspace::vine_announce_sub(addr_hex)
        .expect("vine announce sub should not fail for valid hex")
        .to_string()
}

/// Sans-I/O state machine for vine feed management.
///
/// Tracks followed creators, vine items, and viewed state.
/// The caller feeds events and receives actions to perform.
///
/// **Note:** The `items` map grows without bound as announcements arrive.
/// Callers are responsible for bounding feed size (e.g. evicting old items
/// or limiting the number of followed creators) before it becomes a problem.
#[derive(Debug)]
pub struct VineFeed {
    followed: hashbrown::HashSet<[u8; 16]>,
    items: BTreeMap<(u64, [u8; 32]), VineFeedItem>,
    viewed: hashbrown::HashSet<[u8; 32]>,
}

impl VineFeed {
    pub fn new() -> Self {
        Self {
            followed: hashbrown::HashSet::new(),
            items: BTreeMap::new(),
            viewed: hashbrown::HashSet::new(),
        }
    }

    /// Process an event and return the resulting actions.
    #[must_use]
    pub fn handle_event(&mut self, event: VineEvent) -> Vec<VineAction> {
        match event {
            VineEvent::FollowCreator { address } => {
                if !self.followed.insert(address) {
                    return vec![];
                }
                let addr_hex = hex::encode(address);
                vec![VineAction::Subscribe {
                    key_expr: vine_announce_pattern(&addr_hex),
                }]
            }
            VineEvent::UnfollowCreator { address } => {
                if !self.followed.remove(&address) {
                    return vec![];
                }
                // Purge viewed CIDs for this creator before removing items.
                let mut had_items = false;
                for item in self.items.values() {
                    if item.creator == address {
                        had_items = true;
                        self.viewed.remove(&item.bundle_cid);
                    }
                }
                self.items.retain(|_, item| item.creator != address);
                let addr_hex = hex::encode(address);
                let mut actions = vec![VineAction::Unsubscribe {
                    key_expr: vine_announce_pattern(&addr_hex),
                }];
                if had_items {
                    actions.push(VineAction::FeedUpdated);
                }
                actions
            }
            VineEvent::VineAnnounced { item } => {
                if !self.followed.contains(&item.creator) {
                    return vec![];
                }
                // Evict any existing entry for the same CID (re-announcement
                // with a different timestamp).
                self.items
                    .retain(|_, existing| existing.bundle_cid != item.bundle_cid);
                let key = (item.timestamp, item.bundle_cid);
                self.items.insert(key, item);
                vec![VineAction::FeedUpdated]
            }
            VineEvent::MarkViewed { bundle_cid } => {
                let in_feed = self
                    .items
                    .values()
                    .any(|item| item.bundle_cid == bundle_cid);
                if !in_feed || !self.viewed.insert(bundle_cid) {
                    return vec![];
                }
                vec![VineAction::FeedUpdated]
            }
            VineEvent::MarkAllViewed => {
                let mut changed = false;
                for item in self.items.values() {
                    if self.viewed.insert(item.bundle_cid) {
                        changed = true;
                    }
                }
                if changed {
                    vec![VineAction::FeedUpdated]
                } else {
                    vec![]
                }
            }
        }
    }

    /// Returns unviewed items, newest first.
    pub fn new_items(&self) -> Vec<&VineFeedItem> {
        self.items
            .values()
            .rev()
            .filter(|item| !self.viewed.contains(&item.bundle_cid))
            .collect()
    }

    /// Returns all items, newest first.
    pub fn archive_items(&self) -> Vec<&VineFeedItem> {
        self.items.values().rev().collect()
    }

    /// Returns an iterator over all currently-followed creator addresses.
    ///
    /// Useful for re-issuing `Subscribe` actions after a transport reconnect,
    /// since `FollowCreator` is idempotent and won't re-emit subscriptions
    /// for creators already in the followed set.
    pub fn followed_creators(&self) -> impl Iterator<Item = &[u8; 16]> {
        self.followed.iter()
    }

    /// Returns whether the given address is followed.
    pub fn is_followed(&self, address: &[u8; 16]) -> bool {
        self.followed.contains(address)
    }
}

impl Default for VineFeed {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(creator: [u8; 16], timestamp: u64, bundle_cid: [u8; 32]) -> VineFeedItem {
        VineFeedItem {
            bundle_cid,
            video_cid: [0u8; 32],
            creator,
            timestamp,
            title: None,
            reshare_of: None,
        }
    }

    fn creator_a() -> [u8; 16] {
        let mut addr = [0u8; 16];
        addr[0] = 0xAA;
        addr
    }

    fn cid_from(byte: u8) -> [u8; 32] {
        let mut cid = [0u8; 32];
        cid[0] = byte;
        cid
    }

    #[test]
    fn follow_emits_subscribe() {
        let mut feed = VineFeed::new();
        let actions = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            VineAction::Subscribe { key_expr } => {
                let expected_hex = hex::encode(creator_a());
                assert_eq!(
                    key_expr,
                    &format!("harmony/vines/{expected_hex}/announce/**")
                );
            }
            other => panic!("expected Subscribe, got {:?}", other),
        }
    }

    #[test]
    fn unfollow_emits_unsubscribe() {
        let mut feed = VineFeed::new();
        let _ = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        // No items — only Unsubscribe, no FeedUpdated.
        let actions = feed.handle_event(VineEvent::UnfollowCreator {
            address: creator_a(),
        });
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            VineAction::Unsubscribe { key_expr } => {
                let expected_hex = hex::encode(creator_a());
                assert_eq!(
                    key_expr,
                    &format!("harmony/vines/{expected_hex}/announce/**")
                );
            }
            other => panic!("expected Unsubscribe, got {:?}", other),
        }
    }

    #[test]
    fn unfollow_with_items_emits_feed_updated() {
        let mut feed = VineFeed::new();
        let _ = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        let _ = feed.handle_event(VineEvent::VineAnnounced {
            item: make_item(creator_a(), 100, cid_from(1)),
        });
        let actions = feed.handle_event(VineEvent::UnfollowCreator {
            address: creator_a(),
        });
        assert_eq!(actions.len(), 2);
        assert!(matches!(actions[0], VineAction::Unsubscribe { .. }));
        assert!(matches!(actions[1], VineAction::FeedUpdated));
    }

    #[test]
    fn unfollow_unknown_is_noop() {
        let mut feed = VineFeed::new();
        let actions = feed.handle_event(VineEvent::UnfollowCreator {
            address: creator_a(),
        });
        assert!(actions.is_empty());
    }

    #[test]
    fn vine_announced_adds_to_feed() {
        let mut feed = VineFeed::new();
        let _ = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        let item = make_item(creator_a(), 100, cid_from(1));
        let actions = feed.handle_event(VineEvent::VineAnnounced { item });
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], VineAction::FeedUpdated));
        assert_eq!(feed.new_items().len(), 1);
    }

    #[test]
    fn vine_from_unfollowed_creator_ignored() {
        let mut feed = VineFeed::new();
        let item = make_item(creator_a(), 100, cid_from(1));
        let actions = feed.handle_event(VineEvent::VineAnnounced { item });
        assert!(actions.is_empty());
        assert!(feed.new_items().is_empty());
    }

    #[test]
    fn mark_viewed() {
        let mut feed = VineFeed::new();
        let _ = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        let cid = cid_from(1);
        let _ = feed.handle_event(VineEvent::VineAnnounced {
            item: make_item(creator_a(), 100, cid),
        });
        assert_eq!(feed.new_items().len(), 1);
        let _ = feed.handle_event(VineEvent::MarkViewed { bundle_cid: cid });
        assert!(feed.new_items().is_empty());
        assert_eq!(feed.archive_items().len(), 1);
    }

    #[test]
    fn mark_all_viewed() {
        let mut feed = VineFeed::new();
        let _ = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        let _ = feed.handle_event(VineEvent::VineAnnounced {
            item: make_item(creator_a(), 100, cid_from(1)),
        });
        let _ = feed.handle_event(VineEvent::VineAnnounced {
            item: make_item(creator_a(), 200, cid_from(2)),
        });
        assert_eq!(feed.new_items().len(), 2);
        let _ = feed.handle_event(VineEvent::MarkAllViewed);
        assert!(feed.new_items().is_empty());
        assert_eq!(feed.archive_items().len(), 2);
    }

    #[test]
    fn feed_items_newest_first() {
        let mut feed = VineFeed::new();
        let _ = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        let _ = feed.handle_event(VineEvent::VineAnnounced {
            item: make_item(creator_a(), 100, cid_from(1)),
        });
        let _ = feed.handle_event(VineEvent::VineAnnounced {
            item: make_item(creator_a(), 300, cid_from(3)),
        });
        let _ = feed.handle_event(VineEvent::VineAnnounced {
            item: make_item(creator_a(), 200, cid_from(2)),
        });
        let items = feed.new_items();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].timestamp, 300);
        assert_eq!(items[1].timestamp, 200);
        assert_eq!(items[2].timestamp, 100);
    }

    #[test]
    fn unfollow_clears_viewed_state() {
        let mut feed = VineFeed::new();
        let _ = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        let cid = cid_from(1);
        let _ = feed.handle_event(VineEvent::VineAnnounced {
            item: make_item(creator_a(), 100, cid),
        });
        let _ = feed.handle_event(VineEvent::MarkViewed { bundle_cid: cid });
        assert!(feed.new_items().is_empty());
        // Unfollow should clear viewed state for that creator.
        let _ = feed.handle_event(VineEvent::UnfollowCreator {
            address: creator_a(),
        });
        // Re-follow and re-announce: should appear as new, not already-viewed.
        let _ = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        let _ = feed.handle_event(VineEvent::VineAnnounced {
            item: make_item(creator_a(), 100, cid),
        });
        assert_eq!(feed.new_items().len(), 1);
    }

    #[test]
    fn mark_viewed_duplicate_is_noop() {
        let mut feed = VineFeed::new();
        let _ = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        let cid = cid_from(1);
        let _ = feed.handle_event(VineEvent::VineAnnounced {
            item: make_item(creator_a(), 100, cid),
        });
        let actions1 = feed.handle_event(VineEvent::MarkViewed { bundle_cid: cid });
        assert_eq!(actions1.len(), 1); // FeedUpdated
        let actions2 = feed.handle_event(VineEvent::MarkViewed { bundle_cid: cid });
        assert!(actions2.is_empty()); // No spurious update
    }

    #[test]
    fn reannounce_same_cid_deduplicates() {
        let mut feed = VineFeed::new();
        let _ = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        let cid = cid_from(1);
        let _ = feed.handle_event(VineEvent::VineAnnounced {
            item: make_item(creator_a(), 100, cid),
        });
        // Re-announce same CID with different timestamp.
        let _ = feed.handle_event(VineEvent::VineAnnounced {
            item: make_item(creator_a(), 200, cid),
        });
        // Should have exactly one entry, at the new timestamp.
        let items = feed.archive_items();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].timestamp, 200);
    }

    #[test]
    fn mark_all_viewed_empty_feed_is_noop() {
        let mut feed = VineFeed::new();
        let actions = feed.handle_event(VineEvent::MarkAllViewed);
        assert!(actions.is_empty());
    }

    #[test]
    fn mark_all_viewed_already_viewed_is_noop() {
        let mut feed = VineFeed::new();
        let _ = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        let _ = feed.handle_event(VineEvent::VineAnnounced {
            item: make_item(creator_a(), 100, cid_from(1)),
        });
        let actions1 = feed.handle_event(VineEvent::MarkAllViewed);
        assert_eq!(actions1.len(), 1); // FeedUpdated
        let actions2 = feed.handle_event(VineEvent::MarkAllViewed);
        assert!(actions2.is_empty()); // No spurious update
    }

    #[test]
    fn mark_viewed_unknown_cid_is_noop() {
        let mut feed = VineFeed::new();
        let actions = feed.handle_event(VineEvent::MarkViewed {
            bundle_cid: cid_from(99),
        });
        assert!(actions.is_empty());
    }

    #[test]
    fn followed_creators_returns_all_followed() {
        let mut feed = VineFeed::new();
        assert_eq!(feed.followed_creators().count(), 0);

        let _ = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        let creators: Vec<_> = feed.followed_creators().collect();
        assert_eq!(creators.len(), 1);
        assert!(creators.contains(&&creator_a()));

        // After unfollow, should be empty again.
        let _ = feed.handle_event(VineEvent::UnfollowCreator {
            address: creator_a(),
        });
        assert_eq!(feed.followed_creators().count(), 0);
    }

    #[test]
    fn duplicate_follow_is_idempotent() {
        let mut feed = VineFeed::new();
        let actions1 = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        assert_eq!(actions1.len(), 1); // Subscribe emitted
        let actions2 = feed.handle_event(VineEvent::FollowCreator {
            address: creator_a(),
        });
        assert!(actions2.is_empty()); // No spurious Subscribe
        assert!(feed.is_followed(&creator_a()));
        // Unfollow once should fully remove.
        let actions = feed.handle_event(VineEvent::UnfollowCreator {
            address: creator_a(),
        });
        assert!(!feed.is_followed(&creator_a()));
        assert_eq!(actions.len(), 1); // Unsubscribe only, no items to update
    }
}

//! Subscription matching engine using Zenoh key expression trees.
//!
//! [`SubscriptionTable`] stores subscriptions keyed by key expressions
//! (including wildcards) and efficiently matches incoming messages to all
//! interested subscribers.

use zenoh_keyexpr::key_expr::{keyexpr, OwnedKeyExpr};
use zenoh_keyexpr::keyexpr_tree::box_tree::KeBoxTree;
use zenoh_keyexpr::keyexpr_tree::{IKeyExprTree, IKeyExprTreeMut, IKeyExprTreeNode, IKeyExprTreeNodeMut};

use crate::error::ZenohError;

/// Opaque subscription identifier, returned on subscribe and used to
/// unsubscribe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriptionId(u64);

impl SubscriptionId {
    /// Returns the inner numeric identifier.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Construct from a raw numeric identifier (crate-internal, for testing).
    #[cfg(test)]
    pub(crate) fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

/// A single subscription entry stored in the tree.
#[derive(Debug, Clone)]
struct SubEntry {
    id: SubscriptionId,
}

/// A subscription matching engine backed by a Zenoh KeBoxTree.
///
/// Subscriptions are registered with key expressions (which may contain
/// wildcards like `*` and `**`). When a message arrives on a concrete key,
/// [`matches`](SubscriptionTable::matches) returns all subscription IDs
/// whose key expression intersects with the message key.
pub struct SubscriptionTable {
    tree: KeBoxTree<Vec<SubEntry>>,
    next_id: u64,
}

impl SubscriptionTable {
    /// Create an empty subscription table.
    pub fn new() -> Self {
        Self {
            tree: KeBoxTree::new(),
            next_id: 0,
        }
    }

    /// Register a subscription on the given key expression.
    ///
    /// Returns a [`SubscriptionId`] that can be used to unsubscribe later.
    /// The key expression may contain wildcards (`*` for single chunk,
    /// `**` for multi-chunk).
    pub fn subscribe(&mut self, key_expr: &OwnedKeyExpr) -> SubscriptionId {
        let id = SubscriptionId(self.next_id);
        self.next_id += 1;

        let node = self.tree.node_mut_or_create(key_expr);
        match node.weight_mut() {
            Some(entries) => entries.push(SubEntry { id }),
            None => {
                node.insert_weight(vec![SubEntry { id }]);
            }
        }

        id
    }

    /// Remove a subscription by its ID.
    ///
    /// Returns `Ok(())` if found and removed, or `Err` if not found.
    pub fn unsubscribe(&mut self, id: SubscriptionId) -> Result<(), ZenohError> {
        let mut found = false;
        // Walk all weighted nodes and remove the entry with this ID.
        // KeBoxTree doesn't support targeted removal by value, so we
        // iterate and prune.
        self.tree.prune_where(|node| {
            if let Some(entries) = node.weight_mut() {
                if let Some(pos) = entries.iter().position(|e| e.id == id) {
                    entries.swap_remove(pos);
                    found = true;
                }
                entries.is_empty()
            } else {
                false
            }
        });

        if found {
            Ok(())
        } else {
            Err(ZenohError::SubscriptionNotFound(id.0))
        }
    }

    /// Find all subscription IDs whose key expression intersects with the
    /// given message key.
    ///
    /// This is the core matching operation: for an incoming message on key
    /// `harmony/server/srv1/channel/general/msg`, this will return
    /// subscriptions registered on `harmony/server/srv1/channel/*/msg`,
    /// `harmony/server/*/channel/*/msg`, `harmony/**`, etc.
    pub fn matches(&self, message_key: &keyexpr) -> Vec<SubscriptionId> {
        let mut result = Vec::new();

        // intersecting_nodes returns all nodes whose KE has any overlap with
        // the query. This is a superset of nodes_including (inclusion implies
        // intersection), so a single traversal catches both exact matches and
        // wildcard-to-wildcard matches. Each tree node is visited at most once,
        // and subscription IDs are unique per node, so no dedup is needed.
        for node in self.tree.intersecting_nodes(message_key) {
            if let Some(entries) = node.weight() {
                for entry in entries {
                    result.push(entry.id);
                }
            }
        }

        result
    }

    /// Returns the number of active subscriptions.
    pub fn len(&self) -> usize {
        let mut count = 0;
        for (_ke, entries) in self.tree.key_value_pairs() {
            count += entries.len();
        }
        count
    }

    /// Returns true if there are no subscriptions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for SubscriptionTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyspace;

    fn ke(s: &str) -> OwnedKeyExpr {
        OwnedKeyExpr::autocanonize(s.to_string()).unwrap()
    }

    #[test]
    fn subscribe_and_match_exact() {
        let mut table = SubscriptionTable::new();
        let sub_key = ke("harmony/server/srv1/channel/general/msg");
        let id = table.subscribe(&sub_key);

        let msg_key = keyexpr::new("harmony/server/srv1/channel/general/msg").unwrap();
        let matches = table.matches(msg_key);
        assert_eq!(matches, vec![id]);
    }

    #[test]
    fn wildcard_subscription_matches_concrete_key() {
        let mut table = SubscriptionTable::new();
        let sub = ke("harmony/server/srv1/channel/*/msg");
        let id = table.subscribe(&sub);

        let msg = keyexpr::new("harmony/server/srv1/channel/general/msg").unwrap();
        let matches = table.matches(msg);
        assert_eq!(matches, vec![id]);
    }

    #[test]
    fn double_wildcard_matches_deep_hierarchy() {
        let mut table = SubscriptionTable::new();
        let sub = ke("harmony/**");
        let id = table.subscribe(&sub);

        let msg = keyexpr::new("harmony/server/srv1/channel/general/msg").unwrap();
        let matches = table.matches(msg);
        assert_eq!(matches, vec![id]);
    }

    #[test]
    fn non_matching_subscription_excluded() {
        let mut table = SubscriptionTable::new();
        let sub = ke("harmony/server/srv1/channel/*/msg");
        table.subscribe(&sub);

        let msg = keyexpr::new("harmony/server/srv2/channel/general/msg").unwrap();
        let matches = table.matches(msg);
        assert!(matches.is_empty());
    }

    #[test]
    fn multiple_subscriptions_on_same_key() {
        let mut table = SubscriptionTable::new();
        let sub_key = ke("harmony/server/srv1/channel/*/msg");
        let id1 = table.subscribe(&sub_key);
        let id2 = table.subscribe(&sub_key);

        let msg = keyexpr::new("harmony/server/srv1/channel/general/msg").unwrap();
        let matches = table.matches(msg);
        assert!(matches.contains(&id1));
        assert!(matches.contains(&id2));
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn overlapping_subscriptions_all_match() {
        let mut table = SubscriptionTable::new();
        let id1 = table.subscribe(&ke("harmony/server/srv1/channel/*/msg"));
        let id2 = table.subscribe(&ke("harmony/server/*/channel/general/msg"));
        let id3 = table.subscribe(&ke("harmony/**"));

        let msg = keyexpr::new("harmony/server/srv1/channel/general/msg").unwrap();
        let matches = table.matches(msg);
        assert!(matches.contains(&id1));
        assert!(matches.contains(&id2));
        assert!(matches.contains(&id3));
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn unsubscribe_removes_subscription() {
        let mut table = SubscriptionTable::new();
        let sub_key = ke("harmony/server/srv1/channel/*/msg");
        let id = table.subscribe(&sub_key);

        table.unsubscribe(id).unwrap();

        let msg = keyexpr::new("harmony/server/srv1/channel/general/msg").unwrap();
        let matches = table.matches(msg);
        assert!(matches.is_empty());
    }

    #[test]
    fn unsubscribe_unknown_id_returns_error() {
        let mut table = SubscriptionTable::new();
        let result = table.unsubscribe(SubscriptionId(999));
        assert!(result.is_err());
    }

    #[test]
    fn unsubscribe_one_of_many_on_same_key() {
        let mut table = SubscriptionTable::new();
        let sub_key = ke("harmony/server/srv1/channel/*/msg");
        let id1 = table.subscribe(&sub_key);
        let id2 = table.subscribe(&sub_key);

        table.unsubscribe(id1).unwrap();

        let msg = keyexpr::new("harmony/server/srv1/channel/general/msg").unwrap();
        let matches = table.matches(msg);
        assert_eq!(matches, vec![id2]);
    }

    #[test]
    fn len_tracks_subscription_count() {
        let mut table = SubscriptionTable::new();
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());

        let id1 = table.subscribe(&ke("harmony/server/srv1/channel/*/msg"));
        assert_eq!(table.len(), 1);

        let id2 = table.subscribe(&ke("harmony/presence/*/*"));
        assert_eq!(table.len(), 2);

        table.unsubscribe(id1).unwrap();
        assert_eq!(table.len(), 1);

        table.unsubscribe(id2).unwrap();
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    // ── Integration with keyspace builders ────────────────────────────

    #[test]
    fn keyspace_sub_pattern_matches_keyspace_builder() {
        let mut table = SubscriptionTable::new();
        let sub = keyspace::channel_msg_sub("srv1").unwrap();
        let id = table.subscribe(&sub);

        let msg = keyspace::channel_msg_key("srv1", "general").unwrap();
        let matches = table.matches(&msg);
        assert_eq!(matches, vec![id]);
    }

    #[test]
    fn presence_subscription_matches_presence_key() {
        let mut table = SubscriptionTable::new();
        let sub = keyspace::presence_sub("srv1").unwrap();
        let id = table.subscribe(&sub);

        let p1 = keyspace::presence_key("srv1", "alice").unwrap();
        let p2 = keyspace::presence_key("srv1", "bob").unwrap();
        let p3 = keyspace::presence_key("srv2", "alice").unwrap();

        assert_eq!(table.matches(&p1), vec![id]);
        assert_eq!(table.matches(&p2), vec![id]);
        assert!(table.matches(&p3).is_empty());
    }

    #[test]
    fn geo_s2_l4_subscription_matches_nested_keys() {
        let mut table = SubscriptionTable::new();
        let sub = keyspace::geo_s2_sub_l4("3").unwrap();
        let id = table.subscribe(&sub);

        let g1 = keyspace::geo_s2_key("3", "94", "1234", "5678", "abc").unwrap();
        let g2 = keyspace::geo_s2_key("3", "10", "9999", "8888", "def").unwrap();
        let g3 = keyspace::geo_s2_key("4", "94", "1234", "5678", "abc").unwrap();

        assert_eq!(table.matches(&g1), vec![id]);
        assert_eq!(table.matches(&g2), vec![id]);
        assert!(table.matches(&g3).is_empty());
    }
}

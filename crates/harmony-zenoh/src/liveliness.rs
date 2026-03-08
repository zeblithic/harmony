//! Sans-I/O liveliness token presence tracking.
//!
//! [`LivelinessRouter`] manages liveliness tokens and subscribers for
//! online presence detection without polling. It composes with
//! [`Session`] (peer lifecycle) via caller-driven events — the router
//! has no internal timers.

use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

use zenoh_keyexpr::key_expr::{keyexpr, OwnedKeyExpr};

use crate::error::ZenohError;
use crate::subscription::{SubscriptionId, SubscriptionTable};

/// Opaque liveliness token identifier.
pub type TokenId = u64;

/// Opaque liveliness subscriber identifier.
pub type LivelinessSubscriberId = u64;

/// Inbound events the caller feeds into the router.
#[derive(Debug, Clone)]
pub enum LivelinessEvent {
    /// Remote peer declared a liveliness token.
    TokenDeclared { key_expr: String },
    /// Remote peer undeclared a liveliness token.
    TokenUndeclared { key_expr: String },
    /// Remote peer's session died — bulk-undeclare all their tokens.
    PeerLost,
}

/// Outbound actions the router returns for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LivelinessAction {
    /// Tell the peer we declared a liveliness token.
    SendTokenDeclare { key_expr: String },
    /// Tell the peer we undeclared a liveliness token.
    SendTokenUndeclare { key_expr: String },
    /// Notify a local subscriber that a token appeared.
    TokenAppeared {
        subscriber_id: LivelinessSubscriberId,
        key_expr: String,
    },
    /// Notify a local subscriber that a token disappeared.
    TokenDisappeared {
        subscriber_id: LivelinessSubscriberId,
        key_expr: String,
    },
}

/// A sans-I/O liveliness router managing token declarations, subscriber
/// notifications, and presence queries for a single peer connection.
pub struct LivelinessRouter {
    /// Remote tokens stored for matching against subscribers and queries.
    remote_tokens: SubscriptionTable,
    /// Reverse map: SubscriptionId in remote_tokens → canonical key expression.
    remote_token_keys: HashMap<SubscriptionId, String>,
    /// Reverse map: canonical key expression → SubscriptionId in remote_tokens.
    remote_token_ids: HashMap<String, SubscriptionId>,

    /// Local subscribers for token appear/disappear event fan-out.
    subscribers: SubscriptionTable,
    /// Map: LivelinessSubscriberId → SubscriptionId in subscribers table.
    subscriber_keys: HashMap<LivelinessSubscriberId, SubscriptionId>,
    /// Reverse map: SubscriptionId → LivelinessSubscriberId.
    subscriber_ids: HashMap<SubscriptionId, LivelinessSubscriberId>,

    /// Local tokens declared by this peer.
    local_tokens: HashMap<TokenId, String>,
    /// Refcount per canonical key expression for local tokens.
    /// Wire declare emitted on 0→1, wire undeclare on 1→0.
    local_token_count: HashMap<String, usize>,
    next_token_id: TokenId,
    next_subscriber_id: LivelinessSubscriberId,
}

impl LivelinessRouter {
    /// Create a new empty liveliness router.
    pub fn new() -> Self {
        Self {
            remote_tokens: SubscriptionTable::new(),
            remote_token_keys: HashMap::new(),
            remote_token_ids: HashMap::new(),
            subscribers: SubscriptionTable::new(),
            subscriber_keys: HashMap::new(),
            subscriber_ids: HashMap::new(),
            local_tokens: HashMap::new(),
            local_token_count: HashMap::new(),
            next_token_id: 1,
            next_subscriber_id: 1,
        }
    }

    // ── Token declaration ──────────────────────────────────────────

    /// Declare a local liveliness token on the given key expression.
    ///
    /// Returns the token ID and a `SendTokenDeclare` action for the caller
    /// to send to the peer.
    pub fn declare_token(
        &mut self,
        key_expr: &str,
    ) -> Result<(TokenId, Vec<LivelinessAction>), ZenohError> {
        let owned = OwnedKeyExpr::autocanonize(key_expr.to_string())
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let canonical = owned.to_string();
        let token_id = self.next_token_id;
        self.next_token_id += 1;
        self.local_tokens.insert(token_id, canonical.clone());
        let count = self.local_token_count.entry(canonical.clone()).or_insert(0);
        *count += 1;
        let actions = if *count == 1 {
            vec![LivelinessAction::SendTokenDeclare {
                key_expr: canonical,
            }]
        } else {
            vec![]
        };
        Ok((token_id, actions))
    }

    /// Undeclare a local liveliness token.
    ///
    /// Returns a `SendTokenUndeclare` action for the caller to send to the peer.
    pub fn undeclare_token(
        &mut self,
        token_id: TokenId,
    ) -> Result<Vec<LivelinessAction>, ZenohError> {
        let key_expr = self
            .local_tokens
            .remove(&token_id)
            .ok_or(ZenohError::UnknownTokenId(token_id))?;
        let mut actions = vec![];
        if let Some(count) = self.local_token_count.get_mut(&key_expr) {
            *count -= 1;
            if *count == 0 {
                self.local_token_count.remove(&key_expr);
                actions.push(LivelinessAction::SendTokenUndeclare { key_expr });
            }
        }
        Ok(actions)
    }

    // ── Subscriber management ──────────────────────────────────────

    /// Subscribe to liveliness token changes matching the given key expression.
    ///
    /// Returns the subscriber ID and `TokenAppeared` actions for any remote
    /// tokens that already match (history delivery).
    pub fn subscribe(
        &mut self,
        key_expr: &str,
    ) -> Result<(LivelinessSubscriberId, Vec<LivelinessAction>), ZenohError> {
        let owned = OwnedKeyExpr::autocanonize(key_expr.to_string())
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;

        let sub_table_id = self.subscribers.subscribe(&owned);
        let sub_id = self.next_subscriber_id;
        self.next_subscriber_id += 1;
        self.subscriber_keys.insert(sub_id, sub_table_id);
        self.subscriber_ids.insert(sub_table_id, sub_id);

        // History: emit TokenAppeared for each existing remote token that matches.
        let ke =
            keyexpr::new(owned.as_str()).map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let token_matches = self.remote_tokens.matches(ke);
        let mut actions = Vec::new();
        for token_table_id in token_matches {
            if let Some(token_key) = self.remote_token_keys.get(&token_table_id) {
                actions.push(LivelinessAction::TokenAppeared {
                    subscriber_id: sub_id,
                    key_expr: token_key.clone(),
                });
            }
        }

        Ok((sub_id, actions))
    }

    /// Unsubscribe a liveliness subscriber.
    pub fn unsubscribe(&mut self, subscriber_id: LivelinessSubscriberId) -> Result<(), ZenohError> {
        let sub_table_id = self
            .subscriber_keys
            .remove(&subscriber_id)
            .ok_or(ZenohError::UnknownSubscriptionId(subscriber_id))?;
        self.subscriber_ids.remove(&sub_table_id);
        self.subscribers.unsubscribe(sub_table_id)?;
        Ok(())
    }

    // ── Event handling ─────────────────────────────────────────────

    /// Process an inbound liveliness event and return actions for the caller.
    pub fn handle_event(
        &mut self,
        event: LivelinessEvent,
    ) -> Result<Vec<LivelinessAction>, ZenohError> {
        match event {
            LivelinessEvent::TokenDeclared { key_expr } => self.handle_token_declared(key_expr),
            LivelinessEvent::TokenUndeclared { key_expr } => self.handle_token_undeclared(key_expr),
            LivelinessEvent::PeerLost => Ok(self.handle_peer_lost()),
        }
    }

    fn handle_token_declared(
        &mut self,
        key_expr: String,
    ) -> Result<Vec<LivelinessAction>, ZenohError> {
        let owned = OwnedKeyExpr::autocanonize(key_expr)
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let canonical = owned.to_string();

        // Idempotent: if this exact key is already tracked, skip.
        if self.remote_token_ids.contains_key(&canonical) {
            return Ok(vec![]);
        }

        let sub_id = self.remote_tokens.subscribe(&owned);
        self.remote_token_keys.insert(sub_id, canonical.clone());
        self.remote_token_ids.insert(canonical.clone(), sub_id);

        // Notify matching local subscribers.
        self.notify_subscribers_appeared(&canonical)
    }

    fn handle_token_undeclared(
        &mut self,
        key_expr: String,
    ) -> Result<Vec<LivelinessAction>, ZenohError> {
        let owned = OwnedKeyExpr::autocanonize(key_expr)
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let canonical = owned.to_string();

        // Find and remove the token.
        let Some(table_id) = self.remote_token_ids.remove(&canonical) else {
            return Ok(vec![]); // Unknown token — no-op.
        };

        self.remote_token_keys.remove(&table_id);
        let _ = self.remote_tokens.unsubscribe(table_id);

        self.notify_subscribers_disappeared(&canonical)
    }

    fn handle_peer_lost(&mut self) -> Vec<LivelinessAction> {
        let token_keys: Vec<(SubscriptionId, String)> = self.remote_token_keys.drain().collect();
        self.remote_token_ids.clear();
        let mut actions = Vec::new();

        for (table_id, key_expr) in token_keys {
            let _ = self.remote_tokens.unsubscribe(table_id);
            if let Ok(disappeared) = self.notify_subscribers_disappeared(&key_expr) {
                actions.extend(disappeared);
            }
        }

        actions
    }

    /// Emit TokenAppeared for each local subscriber matching the key expression.
    fn notify_subscribers_appeared(
        &self,
        key_expr: &str,
    ) -> Result<Vec<LivelinessAction>, ZenohError> {
        let ke = keyexpr::new(key_expr).map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let matches = self.subscribers.matches(ke);
        let mut actions = Vec::new();
        for sub_table_id in matches {
            if let Some(&sub_id) = self.subscriber_ids.get(&sub_table_id) {
                actions.push(LivelinessAction::TokenAppeared {
                    subscriber_id: sub_id,
                    key_expr: key_expr.to_string(),
                });
            }
        }
        Ok(actions)
    }

    /// Emit TokenDisappeared for each local subscriber matching the key expression.
    fn notify_subscribers_disappeared(
        &self,
        key_expr: &str,
    ) -> Result<Vec<LivelinessAction>, ZenohError> {
        let ke = keyexpr::new(key_expr).map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let matches = self.subscribers.matches(ke);
        let mut actions = Vec::new();
        for sub_table_id in matches {
            if let Some(&sub_id) = self.subscriber_ids.get(&sub_table_id) {
                actions.push(LivelinessAction::TokenDisappeared {
                    subscriber_id: sub_id,
                    key_expr: key_expr.to_string(),
                });
            }
        }
        Ok(actions)
    }

    // ── Query ─────────────────────────────────────────────────────

    /// Query currently alive remote tokens matching the given key expression.
    ///
    /// Returns the key expressions of all matching tokens.
    pub fn query(&self, key_expr: &str) -> Result<Vec<String>, ZenohError> {
        let owned = OwnedKeyExpr::autocanonize(key_expr.to_string())
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let ke =
            keyexpr::new(owned.as_str()).map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let token_matches = self.remote_tokens.matches(ke);
        let mut results = Vec::new();
        for token_table_id in token_matches {
            if let Some(token_key) = self.remote_token_keys.get(&token_table_id) {
                results.push(token_key.clone());
            }
        }
        Ok(results)
    }
}

impl Default for LivelinessRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_router_is_empty() {
        let router = LivelinessRouter::new();
        assert!(router.local_tokens.is_empty());
        assert!(router.remote_token_keys.is_empty());
        assert!(router.subscriber_keys.is_empty());
    }

    // ── declare_token / undeclare_token ──────────────────────────────

    #[test]
    fn declare_token_returns_id_and_send_action() {
        let mut router = LivelinessRouter::new();
        let (token_id, actions) = router.declare_token("harmony/presence/srv1/alice").unwrap();
        assert_eq!(token_id, 1);
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            LivelinessAction::SendTokenDeclare {
                key_expr: "harmony/presence/srv1/alice".into(),
            }
        );
    }

    #[test]
    fn declare_token_ids_are_monotonic() {
        let mut router = LivelinessRouter::new();
        let (id1, _) = router.declare_token("harmony/presence/srv1/alice").unwrap();
        let (id2, _) = router.declare_token("harmony/presence/srv1/bob").unwrap();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn declare_token_rejects_invalid_key_expr() {
        let mut router = LivelinessRouter::new();
        let result = router.declare_token("");
        assert!(matches!(result, Err(ZenohError::InvalidKeyExpr(_))));
    }

    #[test]
    fn undeclare_token_emits_send_undeclare() {
        let mut router = LivelinessRouter::new();
        let (token_id, _) = router.declare_token("harmony/presence/srv1/alice").unwrap();
        let actions = router.undeclare_token(token_id).unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            LivelinessAction::SendTokenUndeclare {
                key_expr: "harmony/presence/srv1/alice".into(),
            }
        );
    }

    #[test]
    fn undeclare_unknown_token_fails() {
        let mut router = LivelinessRouter::new();
        let result = router.undeclare_token(999);
        assert!(matches!(result, Err(ZenohError::UnknownTokenId(999))));
    }

    #[test]
    fn duplicate_local_token_suppresses_wire_declare() {
        let mut router = LivelinessRouter::new();
        let (id1, actions1) = router.declare_token("harmony/presence/srv1/alice").unwrap();
        assert_eq!(actions1.len(), 1); // First: wire declare
        let (id2, actions2) = router.declare_token("harmony/presence/srv1/alice").unwrap();
        assert!(actions2.is_empty()); // Second: suppressed
        assert_ne!(id1, id2); // Distinct token IDs

        // Undeclare first: no wire undeclare (refcount 2→1)
        let actions3 = router.undeclare_token(id1).unwrap();
        assert!(actions3.is_empty());

        // Undeclare last: wire undeclare emitted (refcount 1→0)
        let actions4 = router.undeclare_token(id2).unwrap();
        assert_eq!(actions4.len(), 1);
        assert_eq!(
            actions4[0],
            LivelinessAction::SendTokenUndeclare {
                key_expr: "harmony/presence/srv1/alice".into(),
            }
        );
    }

    // ── subscribe / unsubscribe ──────────────────────────────────────

    #[test]
    fn subscribe_returns_id() {
        let mut router = LivelinessRouter::new();
        let (sub_id, actions) = router.subscribe("harmony/presence/srv1/*").unwrap();
        assert_eq!(sub_id, 1);
        assert!(actions.is_empty());
    }

    #[test]
    fn subscribe_rejects_invalid_key_expr() {
        let mut router = LivelinessRouter::new();
        let result = router.subscribe("");
        assert!(matches!(result, Err(ZenohError::InvalidKeyExpr(_))));
    }

    #[test]
    fn unsubscribe_unknown_id_fails() {
        let mut router = LivelinessRouter::new();
        let result = router.unsubscribe(999);
        assert!(matches!(
            result,
            Err(ZenohError::UnknownSubscriptionId(999))
        ));
    }

    // ── handle_event ─────────────────────────────────────────────────

    #[test]
    fn subscribe_delivers_existing_remote_tokens() {
        let mut router = LivelinessRouter::new();
        router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        let (sub_id, actions) = router.subscribe("harmony/presence/srv1/*").unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            LivelinessAction::TokenAppeared {
                subscriber_id: sub_id,
                key_expr: "harmony/presence/srv1/alice".into(),
            }
        );
    }

    #[test]
    fn unsubscribe_removes_subscriber() {
        let mut router = LivelinessRouter::new();
        let (sub_id, _) = router.subscribe("harmony/presence/srv1/*").unwrap();
        router.unsubscribe(sub_id).unwrap();
        let actions = router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        let appeared: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, LivelinessAction::TokenAppeared { .. }))
            .collect();
        assert!(appeared.is_empty());
    }

    #[test]
    fn remote_token_declared_notifies_matching_subscribers() {
        let mut router = LivelinessRouter::new();
        let (sub_id, _) = router.subscribe("harmony/presence/srv1/*").unwrap();
        let actions = router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            LivelinessAction::TokenAppeared {
                subscriber_id: sub_id,
                key_expr: "harmony/presence/srv1/alice".into(),
            }
        );
    }

    #[test]
    fn remote_token_declared_no_matching_subscribers() {
        let mut router = LivelinessRouter::new();
        router.subscribe("harmony/presence/srv2/*").unwrap();
        let actions = router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        assert!(actions.is_empty());
    }

    #[test]
    fn remote_token_declared_fans_out_to_multiple_subscribers() {
        let mut router = LivelinessRouter::new();
        let (sub1, _) = router.subscribe("harmony/presence/srv1/*").unwrap();
        let (sub2, _) = router.subscribe("harmony/presence/**").unwrap();
        let actions = router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        assert_eq!(actions.len(), 2);
        let sub_ids: Vec<LivelinessSubscriberId> = actions
            .iter()
            .filter_map(|a| match a {
                LivelinessAction::TokenAppeared { subscriber_id, .. } => Some(*subscriber_id),
                _ => None,
            })
            .collect();
        assert!(sub_ids.contains(&sub1));
        assert!(sub_ids.contains(&sub2));
    }

    #[test]
    fn remote_token_undeclared_notifies_matching_subscribers() {
        let mut router = LivelinessRouter::new();
        let (sub_id, _) = router.subscribe("harmony/presence/srv1/*").unwrap();
        router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        let actions = router
            .handle_event(LivelinessEvent::TokenUndeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            LivelinessAction::TokenDisappeared {
                subscriber_id: sub_id,
                key_expr: "harmony/presence/srv1/alice".into(),
            }
        );
    }

    #[test]
    fn remote_token_undeclared_unknown_is_noop() {
        let mut router = LivelinessRouter::new();
        router.subscribe("harmony/presence/srv1/*").unwrap();
        let actions = router
            .handle_event(LivelinessEvent::TokenUndeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        assert!(actions.is_empty());
    }

    #[test]
    fn duplicate_remote_token_declare_is_idempotent() {
        let mut router = LivelinessRouter::new();
        let (_sub_id, _) = router.subscribe("harmony/presence/srv1/*").unwrap();
        let actions1 = router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        assert_eq!(actions1.len(), 1);
        let actions2 = router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        assert!(actions2.is_empty());
    }

    #[test]
    fn peer_lost_bulk_undeclares_all_remote_tokens() {
        let mut router = LivelinessRouter::new();
        let (_sub_id, _) = router.subscribe("harmony/presence/**").unwrap();
        router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/bob".into(),
            })
            .unwrap();

        let actions = router.handle_event(LivelinessEvent::PeerLost).unwrap();

        let disappeared: Vec<String> = actions
            .iter()
            .filter_map(|a| match a {
                LivelinessAction::TokenDisappeared { key_expr, .. } => Some(key_expr.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(disappeared.len(), 2);
        assert!(disappeared.contains(&"harmony/presence/srv1/alice".to_string()));
        assert!(disappeared.contains(&"harmony/presence/srv1/bob".to_string()));
    }

    #[test]
    fn peer_lost_with_no_remote_tokens_is_noop() {
        let mut router = LivelinessRouter::new();
        router.subscribe("harmony/presence/**").unwrap();
        let actions = router.handle_event(LivelinessEvent::PeerLost).unwrap();
        assert!(actions.is_empty());
    }

    // ── query ────────────────────────────────────────────────────────

    #[test]
    fn query_returns_matching_remote_tokens() {
        let mut router = LivelinessRouter::new();
        router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/bob".into(),
            })
            .unwrap();
        router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv2/carol".into(),
            })
            .unwrap();

        let results = router.query("harmony/presence/srv1/*").unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&"harmony/presence/srv1/alice".to_string()));
        assert!(results.contains(&"harmony/presence/srv1/bob".to_string()));
    }

    #[test]
    fn query_with_no_matching_tokens_returns_empty() {
        let mut router = LivelinessRouter::new();
        router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        let results = router.query("harmony/presence/srv2/*").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn query_rejects_invalid_key_expr() {
        let router = LivelinessRouter::new();
        let result = router.query("");
        assert!(matches!(result, Err(ZenohError::InvalidKeyExpr(_))));
    }

    #[test]
    fn query_after_peer_lost_returns_empty() {
        let mut router = LivelinessRouter::new();
        router
            .handle_event(LivelinessEvent::TokenDeclared {
                key_expr: "harmony/presence/srv1/alice".into(),
            })
            .unwrap();
        router.handle_event(LivelinessEvent::PeerLost).unwrap();
        let results = router.query("harmony/presence/**").unwrap();
        assert!(results.is_empty());
    }
}

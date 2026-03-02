//! Sans-I/O pub/sub routing layer for peer-to-peer message dispatch.
//!
//! [`PubSubRouter`] manages publisher and subscriber declarations,
//! interest-based write-side filtering, and inbound message dispatch.
//! It composes with [`Session`] (peer lifecycle) and [`SubscriptionTable`]
//! (key expression matching) without owning either.

use std::collections::HashMap;
use std::sync::Arc;

use zenoh_keyexpr::key_expr::{keyexpr, OwnedKeyExpr};

use crate::error::ZenohError;
use crate::session::{ExprId, Session, SessionAction};
use crate::subscription::{SubscriptionId, SubscriptionTable};

/// Opaque publisher identifier.
pub type PublisherId = u64;

/// Inbound events the caller feeds into the router.
#[derive(Debug, Clone)]
pub enum PubSubEvent {
    /// Peer declared subscriber interest on a key expression.
    SubscriberDeclared { key_expr: String },
    /// Peer undeclared subscriber interest.
    SubscriberUndeclared { key_expr: String },
    /// Inbound message from peer (already decrypted).
    MessageReceived { expr_id: ExprId, payload: Vec<u8> },
}

/// Outbound actions the router returns for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PubSubAction {
    /// Tell the peer we're subscribing to a key expression.
    SendSubscriberDeclare { key_expr: String },
    /// Tell the peer we're unsubscribing.
    SendSubscriberUndeclare { key_expr: String },
    /// Send a message to the peer (caller encrypts via HarmonyEnvelope).
    SendMessage { expr_id: ExprId, payload: Vec<u8> },
    /// Deliver a received message to a local subscriber.
    ///
    /// Payload is reference-counted to avoid cloning when fanning out
    /// to multiple matching subscribers.
    Deliver {
        subscription_id: SubscriptionId,
        key_expr: String,
        payload: Arc<[u8]>,
    },
    /// Propagate a session-level action (resource declare/undeclare).
    Session(SessionAction),
}

/// A sans-I/O pub/sub router managing publisher/subscriber declarations
/// and message dispatch for a single peer-to-peer connection.
pub struct PubSubRouter {
    subscriptions: SubscriptionTable,
    /// Maps each local subscription ID to its canonical key expression.
    sub_key_exprs: HashMap<SubscriptionId, String>,
    /// Tracks how many local subscriptions exist per canonical key expression.
    /// `SendSubscriberDeclare` is emitted only on 0→1, `SendSubscriberUndeclare` only on 1→0.
    local_interest_count: HashMap<String, usize>,
    remote_interest: SubscriptionTable,
    /// Maps canonical key expression → SubscriptionId in the remote interest table.
    remote_interest_ids: HashMap<String, SubscriptionId>,
    publishers: HashMap<PublisherId, ExprId>,
    next_publisher_id: PublisherId,
}

impl PubSubRouter {
    /// Create a new empty router.
    pub fn new() -> Self {
        Self {
            subscriptions: SubscriptionTable::new(),
            sub_key_exprs: HashMap::new(),
            local_interest_count: HashMap::new(),
            remote_interest: SubscriptionTable::new(),
            remote_interest_ids: HashMap::new(),
            publishers: HashMap::new(),
            next_publisher_id: 1,
        }
    }

    /// Subscribe to a key expression.
    ///
    /// Registers in local SubscriptionTable and emits `SendSubscriberDeclare`
    /// to propagate interest to the peer. If another local subscription already
    /// covers this key expression, the declare is suppressed (refcounted).
    pub fn subscribe(
        &mut self,
        key_expr: &str,
    ) -> Result<(SubscriptionId, Vec<PubSubAction>), ZenohError> {
        let owned = OwnedKeyExpr::autocanonize(key_expr.to_string())
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let canonical = owned.to_string();
        let sub_id = self.subscriptions.subscribe(&owned);
        self.sub_key_exprs.insert(sub_id, canonical.clone());
        let count = self.local_interest_count.entry(canonical.clone()).or_insert(0);
        *count += 1;
        let actions = if *count == 1 {
            vec![PubSubAction::SendSubscriberDeclare { key_expr: canonical }]
        } else {
            vec![]
        };
        Ok((sub_id, actions))
    }

    /// Unsubscribe by subscription ID.
    ///
    /// Removes from local SubscriptionTable and emits `SendSubscriberUndeclare`
    /// only when the last local subscription for this key expression is removed.
    pub fn unsubscribe(
        &mut self,
        sub_id: SubscriptionId,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        let key_expr = self
            .sub_key_exprs
            .remove(&sub_id)
            .ok_or(ZenohError::UnknownSubscriptionId(sub_id.as_u64()))?;
        self.subscriptions.unsubscribe(sub_id)?;
        let mut actions = vec![];
        if let Some(count) = self.local_interest_count.get_mut(&key_expr) {
            *count -= 1;
            if *count == 0 {
                self.local_interest_count.remove(&key_expr);
                actions.push(PubSubAction::SendSubscriberUndeclare { key_expr });
            }
        }
        Ok(actions)
    }

    /// Declare a publisher on the given key expression.
    ///
    /// Validates the key expression, allocates a resource on the session, and
    /// stores the publisher-to-resource mapping. Returns the new `PublisherId`
    /// and any session-level actions (the caller must execute them).
    pub fn declare_publisher(
        &mut self,
        key_expr: String,
        session: &mut Session,
    ) -> Result<(PublisherId, Vec<PubSubAction>), ZenohError> {
        // Validate early so we never allocate a session resource for an invalid key.
        let _owned = OwnedKeyExpr::autocanonize(key_expr.clone())
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let (expr_id, session_actions) = session.declare_resource(key_expr)?;
        let pub_id = self.next_publisher_id;
        self.next_publisher_id += 1;
        self.publishers.insert(pub_id, expr_id);
        let actions = session_actions.into_iter().map(PubSubAction::Session).collect();
        Ok((pub_id, actions))
    }

    /// Undeclare a previously declared publisher.
    ///
    /// Removes the publisher from the router and undeclares its resource
    /// on the session.
    pub fn undeclare_publisher(
        &mut self,
        pub_id: PublisherId,
        session: &mut Session,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        let expr_id = self
            .publishers
            .remove(&pub_id)
            .ok_or(ZenohError::UnknownPublisherId(pub_id))?;
        let session_actions = session.undeclare_resource(expr_id)?;
        let actions = session_actions.into_iter().map(PubSubAction::Session).collect();
        Ok(actions)
    }

    /// Publish a message through the given publisher.
    ///
    /// Resolves the publisher's key expression and checks the remote interest
    /// table. If no remote subscriber is interested, returns an empty action
    /// list (write-side filtering). Otherwise emits a `SendMessage`.
    pub fn publish(
        &self,
        pub_id: PublisherId,
        payload: Vec<u8>,
        session: &Session,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        let expr_id = self
            .publishers
            .get(&pub_id)
            .copied()
            .ok_or(ZenohError::UnknownPublisherId(pub_id))?;

        let key_expr_str = session
            .resolve_local(expr_id)
            .ok_or(ZenohError::UnknownExprId(expr_id))?;

        let ke = keyexpr::new(key_expr_str)
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;

        let matches = self.remote_interest.matches(ke);
        if matches.is_empty() {
            return Ok(vec![]);
        }

        Ok(vec![PubSubAction::SendMessage { expr_id, payload }])
    }

    /// Process an inbound pub/sub event and return actions for the caller.
    pub fn handle_event(
        &mut self,
        event: PubSubEvent,
        session: &Session,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        match event {
            PubSubEvent::SubscriberDeclared { key_expr } => {
                self.handle_subscriber_declared(key_expr)
            }
            PubSubEvent::SubscriberUndeclared { key_expr } => {
                self.handle_subscriber_undeclared(key_expr)
            }
            PubSubEvent::MessageReceived { expr_id, payload } => {
                self.handle_message_received(expr_id, payload, session)
            }
        }
    }

    fn handle_message_received(
        &self,
        expr_id: ExprId,
        payload: Vec<u8>,
        session: &Session,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        let key_expr_str = session
            .resolve_remote(expr_id)
            .ok_or(ZenohError::UnknownExprId(expr_id))?;
        let key = keyexpr::new(key_expr_str)
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let matches = self.subscriptions.matches(key);

        if matches.is_empty() {
            return Ok(vec![]);
        }

        let payload: Arc<[u8]> = Arc::from(payload);
        Ok(matches
            .into_iter()
            .map(|sub_id| PubSubAction::Deliver {
                subscription_id: sub_id,
                key_expr: key_expr_str.to_string(),
                payload: Arc::clone(&payload),
            })
            .collect())
    }

    fn handle_subscriber_declared(
        &mut self,
        key_expr: String,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        let owned = OwnedKeyExpr::autocanonize(key_expr)
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let canonical = owned.to_string();
        // Remove previous interest on the same canonical key to avoid orphaned entries.
        if let Some(old_sub_id) = self.remote_interest_ids.remove(&canonical) {
            let _ = self.remote_interest.unsubscribe(old_sub_id);
        }
        let sub_id = self.remote_interest.subscribe(&owned);
        self.remote_interest_ids.insert(canonical, sub_id);
        Ok(vec![])
    }

    fn handle_subscriber_undeclared(
        &mut self,
        key_expr: String,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        let owned = OwnedKeyExpr::autocanonize(key_expr)
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let canonical = owned.to_string();
        if let Some(sub_id) = self.remote_interest_ids.remove(&canonical) {
            self.remote_interest.unsubscribe(sub_id)?;
        }
        Ok(vec![])
    }
}

impl Default for PubSubRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Session, SessionAction, SessionConfig, SessionEvent, SessionState};
    use harmony_identity::PrivateIdentity;
    use rand::rngs::OsRng;

    fn active_session_pair() -> (Session, Session) {
        let mut rng = OsRng;
        let alice_id = PrivateIdentity::generate(&mut rng);
        let bob_id = PrivateIdentity::generate(&mut rng);
        let alice_pub = alice_id.public_identity().clone();
        let bob_pub = bob_id.public_identity().clone();
        let (mut alice, alice_actions) =
            Session::new(alice_id, bob_pub, SessionConfig::default(), 0);
        let (mut bob, bob_actions) = Session::new(bob_id, alice_pub, SessionConfig::default(), 0);
        let alice_proof = match &alice_actions[0] {
            SessionAction::SendHandshake { proof } => proof.clone(),
            _ => panic!("expected SendHandshake"),
        };
        let bob_proof = match &bob_actions[0] {
            SessionAction::SendHandshake { proof } => proof.clone(),
            _ => panic!("expected SendHandshake"),
        };
        bob.handle_event(SessionEvent::HandshakeReceived {
            proof: alice_proof,
        })
        .unwrap();
        alice
            .handle_event(SessionEvent::HandshakeReceived { proof: bob_proof })
            .unwrap();
        assert_eq!(alice.state(), SessionState::Active);
        assert_eq!(bob.state(), SessionState::Active);
        (alice, bob)
    }

    #[test]
    fn declare_publisher_returns_id_and_declares_resource() {
        let (mut alice, _bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        let (pub_id, actions) = router
            .declare_publisher("harmony/server/srv1/channel/general/msg".into(), &mut alice)
            .unwrap();

        assert_eq!(pub_id, 1);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], PubSubAction::Session(SessionAction::SendResourceDeclare { .. })));
        assert!(alice.resolve_local(1).is_some());
    }

    #[test]
    fn undeclare_publisher_removes_resource() {
        let (mut alice, _bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        let (pub_id, _) = router
            .declare_publisher("harmony/server/srv1/channel/general/msg".into(), &mut alice)
            .unwrap();

        let actions = router.undeclare_publisher(pub_id, &mut alice).unwrap();
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], PubSubAction::Session(SessionAction::SendResourceUndeclare { .. })));
        assert!(alice.resolve_local(1).is_none());
    }

    #[test]
    fn publish_with_remote_interest_emits_send_message() {
        let (mut alice, _bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        let (pub_id, _) = router
            .declare_publisher("harmony/server/srv1/channel/general/msg".into(), &mut alice)
            .unwrap();

        // Simulate remote subscriber declaring interest with wildcard
        router
            .handle_event(
                PubSubEvent::SubscriberDeclared {
                    key_expr: "harmony/server/srv1/channel/*/msg".into(),
                },
                &alice,
            )
            .unwrap();

        let actions = router
            .publish(pub_id, b"hello".to_vec(), &alice)
            .unwrap();

        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            PubSubAction::SendMessage {
                expr_id: 1,
                payload: b"hello".to_vec(),
            }
        );
    }

    #[test]
    fn publish_without_remote_interest_emits_nothing() {
        let (mut alice, _bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        let (pub_id, _) = router
            .declare_publisher("harmony/server/srv1/channel/general/msg".into(), &mut alice)
            .unwrap();

        // No subscriber declared — should emit nothing
        let actions = router
            .publish(pub_id, b"hello".to_vec(), &alice)
            .unwrap();

        assert!(actions.is_empty());
    }

    #[test]
    fn publish_unknown_publisher_fails() {
        let (alice, _bob) = active_session_pair();
        let router = PubSubRouter::new();

        let result = router.publish(999, b"hello".to_vec(), &alice);
        assert!(matches!(result, Err(ZenohError::UnknownPublisherId(999))));
    }

    #[test]
    fn subscribe_emits_declare_action() {
        let mut router = PubSubRouter::new();
        let (sub_id, actions) = router
            .subscribe("harmony/server/srv1/channel/*/msg")
            .unwrap();
        assert_eq!(sub_id.as_u64(), 0); // SubscriptionTable starts at 0
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            PubSubAction::SendSubscriberDeclare { key_expr }
                if key_expr == "harmony/server/srv1/channel/*/msg"
        ));
    }

    #[test]
    fn unsubscribe_emits_undeclare_action() {
        let mut router = PubSubRouter::new();
        let (sub_id, _) = router
            .subscribe("harmony/server/srv1/channel/*/msg")
            .unwrap();
        let actions = router.unsubscribe(sub_id).unwrap();
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            PubSubAction::SendSubscriberUndeclare { key_expr }
                if key_expr == "harmony/server/srv1/channel/*/msg"
        ));
    }

    #[test]
    fn unsubscribe_unknown_id_fails() {
        let mut router = PubSubRouter::new();
        let result = router.unsubscribe(SubscriptionId::from_raw(999));
        assert!(matches!(
            result,
            Err(ZenohError::UnknownSubscriptionId(999))
        ));
    }

    #[test]
    fn inbound_message_with_matching_subscriber_delivers() {
        let (mut alice, mut bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        // Alice subscribes to channel messages
        let (sub_id, _) = router
            .subscribe("harmony/server/srv1/channel/*/msg")
            .unwrap();

        // Bob declares a resource (simulated: alice receives ResourceDeclared event)
        let (expr_id, _) = bob
            .declare_resource("harmony/server/srv1/channel/general/msg".into())
            .unwrap();
        alice
            .handle_event(SessionEvent::ResourceDeclared {
                expr_id,
                key_expr: "harmony/server/srv1/channel/general/msg".into(),
            })
            .unwrap();

        // Inbound message from bob
        let actions = router
            .handle_event(
                PubSubEvent::MessageReceived {
                    expr_id,
                    payload: b"hello".to_vec(),
                },
                &alice,
            )
            .unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            PubSubAction::Deliver {
                subscription_id: sub_id,
                key_expr: "harmony/server/srv1/channel/general/msg".into(),
                payload: Arc::from(b"hello".as_slice()),
            }
        );
    }

    #[test]
    fn inbound_message_with_no_matching_subscriber_drops() {
        let (mut alice, mut bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        // No local subscriptions

        // Bob declares a resource
        let (expr_id, _) = bob
            .declare_resource("harmony/server/srv1/channel/general/msg".into())
            .unwrap();
        alice
            .handle_event(SessionEvent::ResourceDeclared {
                expr_id,
                key_expr: "harmony/server/srv1/channel/general/msg".into(),
            })
            .unwrap();

        let actions = router
            .handle_event(
                PubSubEvent::MessageReceived {
                    expr_id,
                    payload: b"hello".to_vec(),
                },
                &alice,
            )
            .unwrap();
        assert!(actions.is_empty());
    }

    #[test]
    fn inbound_message_with_multiple_subscribers_delivers_to_all() {
        let (mut alice, mut bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        // Two overlapping subscriptions
        let (sub1, _) = router
            .subscribe("harmony/server/srv1/channel/*/msg")
            .unwrap();
        let (sub2, _) = router.subscribe("harmony/**").unwrap();

        // Bob declares a resource
        let (expr_id, _) = bob
            .declare_resource("harmony/server/srv1/channel/general/msg".into())
            .unwrap();
        alice
            .handle_event(SessionEvent::ResourceDeclared {
                expr_id,
                key_expr: "harmony/server/srv1/channel/general/msg".into(),
            })
            .unwrap();

        let actions = router
            .handle_event(
                PubSubEvent::MessageReceived {
                    expr_id,
                    payload: b"hello".to_vec(),
                },
                &alice,
            )
            .unwrap();
        assert_eq!(actions.len(), 2);

        let sub_ids: Vec<SubscriptionId> = actions
            .iter()
            .filter_map(|a| match a {
                PubSubAction::Deliver {
                    subscription_id, ..
                } => Some(*subscription_id),
                _ => None,
            })
            .collect();
        assert!(sub_ids.contains(&sub1));
        assert!(sub_ids.contains(&sub2));
    }

    #[test]
    fn duplicate_remote_interest_does_not_leak() {
        let (mut alice, _bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        let (pub_id, _) = router
            .declare_publisher("harmony/server/srv1/channel/general/msg".into(), &mut alice)
            .unwrap();

        let ke = "harmony/server/srv1/channel/*/msg";

        // Declare interest twice on the same key expression
        router
            .handle_event(PubSubEvent::SubscriberDeclared { key_expr: ke.into() }, &alice)
            .unwrap();
        router
            .handle_event(PubSubEvent::SubscriberDeclared { key_expr: ke.into() }, &alice)
            .unwrap();

        // Should still match (second declare replaced first)
        let actions = router.publish(pub_id, b"msg".to_vec(), &alice).unwrap();
        assert_eq!(actions.len(), 1);

        // Single undeclare should clear interest — no orphaned entry
        router
            .handle_event(PubSubEvent::SubscriberUndeclared { key_expr: ke.into() }, &alice)
            .unwrap();

        let actions = router.publish(pub_id, b"msg".to_vec(), &alice).unwrap();
        assert!(actions.is_empty());
    }

    #[test]
    fn remote_interest_declare_undeclare_lifecycle() {
        let (mut alice, _bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        let (pub_id, _) = router
            .declare_publisher("harmony/server/srv1/channel/general/msg".into(), &mut alice)
            .unwrap();

        // No remote interest yet — publish should emit nothing
        let actions = router.publish(pub_id, b"msg1".to_vec(), &alice).unwrap();
        assert!(actions.is_empty());

        // Peer declares interest
        router
            .handle_event(
                PubSubEvent::SubscriberDeclared {
                    key_expr: "harmony/server/srv1/channel/*/msg".into(),
                },
                &alice,
            )
            .unwrap();

        // Now publish should emit SendMessage
        let actions = router.publish(pub_id, b"msg2".to_vec(), &alice).unwrap();
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], PubSubAction::SendMessage { .. }));

        // Peer undeclares interest
        router
            .handle_event(
                PubSubEvent::SubscriberUndeclared {
                    key_expr: "harmony/server/srv1/channel/*/msg".into(),
                },
                &alice,
            )
            .unwrap();

        // Publish should emit nothing again
        let actions = router.publish(pub_id, b"msg3".to_vec(), &alice).unwrap();
        assert!(actions.is_empty());
    }

    #[test]
    fn duplicate_subscribe_refcounts_interest_declaration() {
        let mut router = PubSubRouter::new();
        let ke = "harmony/server/srv1/channel/*/msg";

        // First subscribe emits declare
        let (sub1, actions1) = router.subscribe(ke).unwrap();
        assert_eq!(actions1.len(), 1);
        assert!(matches!(
            &actions1[0],
            PubSubAction::SendSubscriberDeclare { .. }
        ));

        // Second subscribe on same key: no declare (refcount 1→2)
        let (sub2, actions2) = router.subscribe(ke).unwrap();
        assert!(actions2.is_empty());

        // Unsubscribe first: no undeclare (refcount 2→1)
        let actions3 = router.unsubscribe(sub1).unwrap();
        assert!(actions3.is_empty());

        // Unsubscribe last: undeclare emitted (refcount 1→0)
        let actions4 = router.unsubscribe(sub2).unwrap();
        assert_eq!(actions4.len(), 1);
        assert!(matches!(
            &actions4[0],
            PubSubAction::SendSubscriberUndeclare { .. }
        ));
    }

    #[test]
    fn remote_interest_canonical_key_roundtrip() {
        let (mut alice, _bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        let (pub_id, _) = router
            .declare_publisher("harmony/server/srv1/channel/general/msg".into(), &mut alice)
            .unwrap();

        // Declare with non-canonical form (Zenoh canonicalizes $* → *)
        router
            .handle_event(
                PubSubEvent::SubscriberDeclared {
                    key_expr: "harmony/server/srv1/channel/$*/msg".into(),
                },
                &alice,
            )
            .unwrap();

        // Should match via the canonical form
        let actions = router.publish(pub_id, b"msg".to_vec(), &alice).unwrap();
        assert_eq!(actions.len(), 1);

        // Undeclare with canonical form — should still find the entry
        router
            .handle_event(
                PubSubEvent::SubscriberUndeclared {
                    key_expr: "harmony/server/srv1/channel/*/msg".into(),
                },
                &alice,
            )
            .unwrap();

        // Interest gone
        let actions = router.publish(pub_id, b"msg".to_vec(), &alice).unwrap();
        assert!(actions.is_empty());
    }

    #[test]
    fn declare_publisher_rejects_invalid_key_expression() {
        let (mut alice, _bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        // Empty string is not a valid Zenoh key expression
        let result = router.declare_publisher("".into(), &mut alice);
        assert!(matches!(result, Err(ZenohError::InvalidKeyExpr(_))));

        // No session resource should have been allocated
        assert!(alice.resolve_local(1).is_none());
    }
}

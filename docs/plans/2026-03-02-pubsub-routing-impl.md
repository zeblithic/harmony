# Pub/Sub Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a sans-I/O pub/sub routing layer that manages publisher/subscriber declarations, interest-based write-side filtering, and inbound message dispatch.

**Architecture:** `PubSubRouter` is a thin coordinator that owns a `SubscriptionTable` for local subscriptions, a second `SubscriptionTable` for remote interest tracking, and publisher-to-ExprId mappings. It takes `&mut Session` / `&Session` references when handling events and returns `PubSubAction` variants for the caller to execute.

**Tech Stack:** Rust, harmony-zenoh (session, subscription, keyspace), zenoh-keyexpr

---

### Task 1: Error variants + pubsub module skeleton

Add the two new error variants and create the `pubsub.rs` module with all public types (no logic yet).

**Files:**
- Modify: `crates/harmony-zenoh/src/error.rs:35` (before closing brace)
- Create: `crates/harmony-zenoh/src/pubsub.rs`
- Modify: `crates/harmony-zenoh/src/lib.rs:5` (add module) and `lib.rs:11` (add re-exports)

**Step 1: Add error variants**

In `crates/harmony-zenoh/src/error.rs`, add before the closing `}` on line 36:

```rust
    #[error("unknown publisher ID: {0}")]
    UnknownPublisherId(u64),

    #[error("unknown subscription ID: {0}")]
    UnknownSubscriptionId(u64),
```

**Step 2: Create pubsub.rs with type definitions**

Create `crates/harmony-zenoh/src/pubsub.rs`:

```rust
//! Sans-I/O pub/sub routing layer for peer-to-peer message dispatch.
//!
//! [`PubSubRouter`] manages publisher and subscriber declarations,
//! interest-based write-side filtering, and inbound message dispatch.
//! It composes with [`Session`] (peer lifecycle) and [`SubscriptionTable`]
//! (key expression matching) without owning either.

use std::collections::HashMap;

use zenoh_keyexpr::key_expr::OwnedKeyExpr;

use crate::session::{ExprId, Session};
use crate::subscription::{SubscriptionId, SubscriptionTable};
use crate::ZenohError;

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
    Deliver {
        subscription_id: SubscriptionId,
        key_expr: String,
        payload: Vec<u8>,
    },
}

/// A sans-I/O pub/sub router managing publisher/subscriber declarations
/// and message dispatch for a single peer-to-peer connection.
pub struct PubSubRouter {
    subscriptions: SubscriptionTable,
    sub_key_exprs: HashMap<SubscriptionId, String>,
    remote_interest: SubscriptionTable,
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
            remote_interest: SubscriptionTable::new(),
            remote_interest_ids: HashMap::new(),
            publishers: HashMap::new(),
            next_publisher_id: 1,
        }
    }
}

impl Default for PubSubRouter {
    fn default() -> Self {
        Self::new()
    }
}
```

**Step 3: Wire up lib.rs**

In `crates/harmony-zenoh/src/lib.rs`, add `pub mod pubsub;` after line 4 (the `session` line), and add re-exports:

```rust
pub mod envelope;
pub mod error;
pub mod keyspace;
pub mod pubsub;
pub mod session;
pub mod subscription;

pub use envelope::{HarmonyEnvelope, MessageType, HEADER_SIZE, MIN_ENVELOPE_SIZE, VERSION};
pub use error::ZenohError;
pub use keyspace::keyexpr;
pub use pubsub::{PubSubAction, PubSubEvent, PubSubRouter, PublisherId};
pub use session::{ExprId, Session, SessionAction, SessionConfig, SessionEvent, SessionState};
pub use subscription::{SubscriptionId, SubscriptionTable};
```

**Step 4: Verify it compiles**

Run: `cargo test -p harmony-zenoh --lib`
Expected: All existing tests pass, no compile errors.

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/error.rs crates/harmony-zenoh/src/pubsub.rs crates/harmony-zenoh/src/lib.rs
git commit -m "Add pubsub module skeleton with core types and error variants"
```

---

### Task 2: Publisher declare, undeclare, and publish with interest filtering

Implement the publisher lifecycle: declare (allocates resource on session), undeclare, and publish with write-side filtering via remote interest table.

**Files:**
- Modify: `crates/harmony-zenoh/src/pubsub.rs`

**Step 1: Write the failing tests**

Add a `#[cfg(test)] mod tests` block at the bottom of `pubsub.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Session, SessionAction, SessionConfig, SessionEvent, SessionState};
    use harmony_identity::PrivateIdentity;
    use rand::rngs::OsRng;

    /// Create a session pair and complete the handshake so both are Active.
    fn active_session_pair() -> (Session, Session) {
        let mut rng = OsRng;
        let alice_id = PrivateIdentity::generate(&mut rng);
        let bob_id = PrivateIdentity::generate(&mut rng);
        let alice_pub = alice_id.public_identity().clone();
        let bob_pub = bob_id.public_identity().clone();

        let (mut alice, alice_actions) =
            Session::new(alice_id, bob_pub, SessionConfig::default(), 0);
        let (mut bob, bob_actions) =
            Session::new(bob_id, alice_pub, SessionConfig::default(), 0);

        // Extract handshake proofs
        let alice_proof = match &alice_actions[0] {
            SessionAction::SendHandshake { proof } => proof.clone(),
            _ => panic!("expected SendHandshake"),
        };
        let bob_proof = match &bob_actions[0] {
            SessionAction::SendHandshake { proof } => proof.clone(),
            _ => panic!("expected SendHandshake"),
        };

        // Complete handshake
        bob.handle_event(SessionEvent::HandshakeReceived { proof: alice_proof })
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

        let (pub_id, _actions) = router
            .declare_publisher("harmony/server/srv1/channel/general/msg".into(), &mut alice)
            .unwrap();
        assert_eq!(pub_id, 1);

        // Session should have the resource declared
        assert!(alice.resolve_local(1).is_some());
    }

    #[test]
    fn undeclare_publisher_removes_resource() {
        let (mut alice, _bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        let (pub_id, _) = router
            .declare_publisher("harmony/server/srv1/channel/general/msg".into(), &mut alice)
            .unwrap();
        router.undeclare_publisher(pub_id, &mut alice).unwrap();

        assert!(alice.resolve_local(1).is_none());
    }

    #[test]
    fn publish_with_remote_interest_emits_send_message() {
        let (mut alice, _bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        let (pub_id, _) = router
            .declare_publisher("harmony/server/srv1/channel/general/msg".into(), &mut alice)
            .unwrap();

        // Simulate peer declaring interest
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
        assert!(matches!(&actions[0], PubSubAction::SendMessage { expr_id: 1, .. }));
    }

    #[test]
    fn publish_without_remote_interest_emits_nothing() {
        let (mut alice, _bob) = active_session_pair();
        let mut router = PubSubRouter::new();

        let (pub_id, _) = router
            .declare_publisher("harmony/server/srv1/channel/general/msg".into(), &mut alice)
            .unwrap();

        // No remote interest declared
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
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh session::tests -- --ignored 2>&1; cargo test -p harmony-zenoh pubsub 2>&1`
Expected: 5 compile errors — `declare_publisher`, `undeclare_publisher`, `publish`, `handle_event` methods don't exist yet.

**Step 3: Implement publisher methods and handle_event skeleton**

Add to the `impl PubSubRouter` block in `pubsub.rs`:

```rust
    /// Declare a publisher on a key expression.
    ///
    /// Declares the resource on the session and returns a PublisherId.
    pub fn declare_publisher(
        &mut self,
        key_expr: String,
        session: &mut Session,
    ) -> Result<(PublisherId, Vec<PubSubAction>), ZenohError> {
        let (expr_id, _session_actions) = session.declare_resource(key_expr)?;
        let pub_id = self.next_publisher_id;
        self.next_publisher_id += 1;
        self.publishers.insert(pub_id, expr_id);
        Ok((pub_id, vec![]))
    }

    /// Undeclare a publisher.
    pub fn undeclare_publisher(
        &mut self,
        pub_id: PublisherId,
        session: &mut Session,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        let expr_id = self
            .publishers
            .remove(&pub_id)
            .ok_or(ZenohError::UnknownPublisherId(pub_id))?;
        session.undeclare_resource(expr_id)?;
        Ok(vec![])
    }

    /// Publish a message. Checks remote interest before emitting SendMessage.
    pub fn publish(
        &self,
        pub_id: PublisherId,
        payload: Vec<u8>,
        session: &Session,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        let expr_id = self
            .publishers
            .get(&pub_id)
            .ok_or(ZenohError::UnknownPublisherId(pub_id))?;
        let key_expr = session
            .resolve_local(*expr_id)
            .ok_or(ZenohError::UnknownExprId(*expr_id))?;
        let key = zenoh_keyexpr::key_expr::keyexpr::new(key_expr)
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        if self.remote_interest.matches(key).is_empty() {
            return Ok(vec![]);
        }
        Ok(vec![PubSubAction::SendMessage {
            expr_id: *expr_id,
            payload,
        }])
    }

    /// Process an inbound event and return actions for the caller to execute.
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

    fn handle_subscriber_declared(
        &mut self,
        key_expr: String,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        let owned = OwnedKeyExpr::autocanonize(key_expr.clone())
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let sub_id = self.remote_interest.subscribe(&owned);
        self.remote_interest_ids.insert(key_expr, sub_id);
        Ok(vec![])
    }

    fn handle_subscriber_undeclared(
        &mut self,
        key_expr: String,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        let sub_id = self
            .remote_interest_ids
            .remove(&key_expr)
            .ok_or(ZenohError::UnknownSubscriptionId(0))?;
        self.remote_interest.unsubscribe(sub_id)?;
        Ok(vec![])
    }

    fn handle_message_received(
        &self,
        _expr_id: ExprId,
        _payload: Vec<u8>,
        _session: &Session,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        // Stub — implemented in Task 4
        Ok(vec![])
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-zenoh pubsub`
Expected: 5 tests pass.

**Step 5: Run clippy**

Run: `cargo clippy -p harmony-zenoh -- -D warnings`
Expected: Clean.

**Step 6: Commit**

```bash
git add crates/harmony-zenoh/src/pubsub.rs
git commit -m "Implement publisher declare, undeclare, and publish with interest filtering"
```

---

### Task 3: Subscriber declare and unsubscribe with interest propagation

Implement local subscription lifecycle: subscribe (registers locally + emits `SendSubscriberDeclare`), unsubscribe (removes + emits `SendSubscriberUndeclare`).

**Files:**
- Modify: `crates/harmony-zenoh/src/pubsub.rs`

**Step 1: Write the failing tests**

Add to the existing `mod tests` block in `pubsub.rs`:

```rust
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
```

Note: `SubscriptionId::from_raw(999)` doesn't exist yet. We need a way to construct a `SubscriptionId` from a raw `u64` for testing error paths. Two options:
- Make `SubscriptionId(pub u64)` — exposes internals.
- Add `pub(crate) fn from_raw(id: u64) -> Self` — testable within the crate.

Use `from_raw` — add it to `subscription.rs`.

**Step 2: Add `from_raw` to SubscriptionId**

In `crates/harmony-zenoh/src/subscription.rs`, add after the `as_u64` method (line 22):

```rust
    /// Construct from a raw numeric identifier (crate-internal, for testing).
    pub(crate) fn from_raw(id: u64) -> Self {
        Self(id)
    }
```

**Step 3: Implement subscribe and unsubscribe**

Add to the `impl PubSubRouter` block:

```rust
    /// Subscribe to a key expression.
    ///
    /// Registers in local SubscriptionTable and emits `SendSubscriberDeclare`
    /// to propagate interest to the peer.
    pub fn subscribe(
        &mut self,
        key_expr: &str,
    ) -> Result<(SubscriptionId, Vec<PubSubAction>), ZenohError> {
        let owned = OwnedKeyExpr::autocanonize(key_expr.to_string())
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let sub_id = self.subscriptions.subscribe(&owned);
        self.sub_key_exprs.insert(sub_id, key_expr.to_string());
        Ok((
            sub_id,
            vec![PubSubAction::SendSubscriberDeclare {
                key_expr: key_expr.to_string(),
            }],
        ))
    }

    /// Unsubscribe by subscription ID.
    ///
    /// Removes from local SubscriptionTable and emits `SendSubscriberUndeclare`.
    pub fn unsubscribe(
        &mut self,
        sub_id: SubscriptionId,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        let key_expr = self
            .sub_key_exprs
            .remove(&sub_id)
            .ok_or(ZenohError::UnknownSubscriptionId(sub_id.as_u64()))?;
        self.subscriptions.unsubscribe(sub_id)?;
        Ok(vec![PubSubAction::SendSubscriberUndeclare { key_expr }])
    }
```

**Step 4: Run tests**

Run: `cargo test -p harmony-zenoh pubsub`
Expected: 8 tests pass (5 from Task 2 + 3 new).

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/pubsub.rs crates/harmony-zenoh/src/subscription.rs
git commit -m "Implement subscriber declare, unsubscribe, and interest propagation"
```

---

### Task 4: Inbound message dispatch

Implement `handle_message_received`: resolve ExprId via session, match against local subscriptions, emit `Deliver` per match.

**Files:**
- Modify: `crates/harmony-zenoh/src/pubsub.rs`

**Step 1: Write the failing tests**

Add to the existing `mod tests` block:

```rust
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
        assert!(matches!(
            &actions[0],
            PubSubAction::Deliver {
                subscription_id,
                key_expr,
                payload,
            } if *subscription_id == sub_id
                && key_expr == "harmony/server/srv1/channel/general/msg"
                && payload == b"hello"
        ));
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
                PubSubAction::Deliver { subscription_id, .. } => Some(*subscription_id),
                _ => None,
            })
            .collect();
        assert!(sub_ids.contains(&sub1));
        assert!(sub_ids.contains(&sub2));
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh pubsub`
Expected: 3 new tests fail (message dispatch returns empty vec from stub).

**Step 3: Implement handle_message_received**

Replace the stub in `pubsub.rs`:

```rust
    fn handle_message_received(
        &self,
        expr_id: ExprId,
        payload: Vec<u8>,
        session: &Session,
    ) -> Result<Vec<PubSubAction>, ZenohError> {
        let key_expr_str = session
            .resolve_remote(expr_id)
            .ok_or(ZenohError::UnknownExprId(expr_id))?;
        let key = zenoh_keyexpr::key_expr::keyexpr::new(key_expr_str)
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let matches = self.subscriptions.matches(key);

        if matches.is_empty() {
            return Ok(vec![]);
        }

        Ok(matches
            .into_iter()
            .map(|sub_id| PubSubAction::Deliver {
                subscription_id: sub_id,
                key_expr: key_expr_str.to_string(),
                payload: payload.clone(),
            })
            .collect())
    }
```

**Step 4: Run tests**

Run: `cargo test -p harmony-zenoh pubsub`
Expected: 11 tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/pubsub.rs
git commit -m "Implement inbound message dispatch with subscription matching"
```

---

### Task 5: Remote interest lifecycle test

Add the integration test that verifies the full remote interest declare/undeclare cycle: peer subscribes → publish works → peer unsubscribes → publish filtered.

**Files:**
- Modify: `crates/harmony-zenoh/src/pubsub.rs`

**Step 1: Write the test**

Add to the existing `mod tests` block:

```rust
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
```

**Step 2: Run tests**

Run: `cargo test -p harmony-zenoh pubsub`
Expected: 12 tests pass.

**Step 3: Run full workspace verification**

Run: `cargo test --workspace`
Expected: All tests pass (existing + 12 new pubsub tests).

Run: `cargo clippy --workspace -- -D warnings`
Expected: Clean.

**Step 4: Commit**

```bash
git add crates/harmony-zenoh/src/pubsub.rs
git commit -m "Add remote interest lifecycle integration test"
```

# Liveliness Tracking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement liveliness token presence tracking for online status detection without polling in the harmony-zenoh crate.

**Architecture:** Standalone `LivelinessRouter` sans-I/O state machine using two `SubscriptionTable` instances — one for remote tokens (matched against local subscribers + queried), one for local subscribers (matched when tokens appear/disappear). No internal timers; disconnect detection delegates to Session's existing PeerStale/SessionClosed via a PeerLost event.

**Tech Stack:** Rust, zenoh-keyexpr (OwnedKeyExpr, keyexpr, KeBoxTree via SubscriptionTable)

**Design doc:** `docs/plans/2026-03-02-liveliness-tracking-design.md`

---

### Task 1: Add UnknownTokenId error variant

**Files:**
- Modify: `crates/harmony-zenoh/src/error.rs`

**Step 1: Add the new error variant**

Add to the `ZenohError` enum in `error.rs`, after the existing `UnknownSubscriptionId` variant:

```rust
#[error("unknown token ID: {0}")]
UnknownTokenId(u64),
```

**Step 2: Run tests to verify nothing breaks**

Run: `cargo test -p harmony-zenoh`
Expected: All existing tests pass (no code references the new variant yet).

**Step 3: Commit**

```bash
git add crates/harmony-zenoh/src/error.rs
git commit -m "feat(zenoh): add UnknownTokenId error variant"
```

---

### Task 2: Create liveliness.rs with types and empty struct

**Files:**
- Create: `crates/harmony-zenoh/src/liveliness.rs`
- Modify: `crates/harmony-zenoh/src/lib.rs`

**Step 1: Write the failing test**

Create `crates/harmony-zenoh/src/liveliness.rs` with the type definitions, struct, constructor, and a basic test:

```rust
//! Sans-I/O liveliness token presence tracking.
//!
//! [`LivelinessRouter`] manages liveliness tokens and subscribers for
//! online presence detection without polling. It composes with
//! [`Session`] (peer lifecycle) via caller-driven events — the router
//! has no internal timers.

use std::collections::HashMap;

use zenoh_keyexpr::key_expr::OwnedKeyExpr;

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

    /// Local subscribers for token appear/disappear event fan-out.
    subscribers: SubscriptionTable,
    /// Map: LivelinessSubscriberId → SubscriptionId in subscribers table.
    subscriber_keys: HashMap<LivelinessSubscriberId, SubscriptionId>,

    /// Local tokens declared by this peer.
    local_tokens: HashMap<TokenId, String>,
    next_token_id: TokenId,
    next_subscriber_id: LivelinessSubscriberId,
}

impl LivelinessRouter {
    /// Create a new empty liveliness router.
    pub fn new() -> Self {
        Self {
            remote_tokens: SubscriptionTable::new(),
            remote_token_keys: HashMap::new(),
            subscribers: SubscriptionTable::new(),
            subscriber_keys: HashMap::new(),
            local_tokens: HashMap::new(),
            next_token_id: 1,
            next_subscriber_id: 1,
        }
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
}
```

**Step 2: Add module to lib.rs**

In `crates/harmony-zenoh/src/lib.rs`, add the module declaration and re-exports. After the existing `pub mod subscription;` line, add:

```rust
pub mod liveliness;
```

After the existing `pub use subscription::{SubscriptionId, SubscriptionTable};` line, add:

```rust
pub use liveliness::{
    LivelinessAction, LivelinessEvent, LivelinessRouter, LivelinessSubscriberId, TokenId,
};
```

**Step 3: Run tests to verify they pass**

Run: `cargo test -p harmony-zenoh liveliness`
Expected: 1 test passes (`new_router_is_empty`).

**Step 4: Commit**

```bash
git add crates/harmony-zenoh/src/liveliness.rs crates/harmony-zenoh/src/lib.rs
git commit -m "feat(zenoh): add LivelinessRouter skeleton with types"
```

---

### Task 3: Implement declare_token and undeclare_token

**Files:**
- Modify: `crates/harmony-zenoh/src/liveliness.rs`

**Step 1: Write the failing tests**

Add to the `tests` module in `liveliness.rs`:

```rust
#[test]
fn declare_token_returns_id_and_send_action() {
    let mut router = LivelinessRouter::new();
    let (token_id, actions) = router
        .declare_token("harmony/presence/srv1/alice")
        .unwrap();
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
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh liveliness`
Expected: FAIL — `declare_token` and `undeclare_token` methods don't exist.

**Step 3: Implement declare_token and undeclare_token**

Add these methods to the `impl LivelinessRouter` block:

```rust
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
    Ok((
        token_id,
        vec![LivelinessAction::SendTokenDeclare {
            key_expr: canonical,
        }],
    ))
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
    Ok(vec![LivelinessAction::SendTokenUndeclare { key_expr }])
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-zenoh liveliness`
Expected: All 6 tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/liveliness.rs
git commit -m "feat(zenoh): implement declare_token and undeclare_token"
```

---

### Task 4: Implement subscribe and unsubscribe

**Files:**
- Modify: `crates/harmony-zenoh/src/liveliness.rs`

**Step 1: Write the failing tests**

Add to the `tests` module:

```rust
#[test]
fn subscribe_returns_id() {
    let mut router = LivelinessRouter::new();
    let (sub_id, actions) = router
        .subscribe("harmony/presence/srv1/*")
        .unwrap();
    assert_eq!(sub_id, 1);
    // No remote tokens exist yet, so no TokenAppeared actions
    assert!(actions.is_empty());
}

#[test]
fn subscribe_delivers_existing_remote_tokens() {
    let mut router = LivelinessRouter::new();
    // Remote peer declared a token before we subscribed
    router
        .handle_event(LivelinessEvent::TokenDeclared {
            key_expr: "harmony/presence/srv1/alice".into(),
        })
        .unwrap();
    // Now subscribe — should get TokenAppeared for existing token
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
fn subscribe_rejects_invalid_key_expr() {
    let mut router = LivelinessRouter::new();
    let result = router.subscribe("");
    assert!(matches!(result, Err(ZenohError::InvalidKeyExpr(_))));
}

#[test]
fn unsubscribe_removes_subscriber() {
    let mut router = LivelinessRouter::new();
    let (sub_id, _) = router.subscribe("harmony/presence/srv1/*").unwrap();
    router.unsubscribe(sub_id).unwrap();
    // Subsequent token should not generate events for removed subscriber
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
fn unsubscribe_unknown_id_fails() {
    let mut router = LivelinessRouter::new();
    let result = router.unsubscribe(999);
    assert!(matches!(
        result,
        Err(ZenohError::UnknownSubscriptionId(999))
    ));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh liveliness`
Expected: FAIL — `subscribe`, `unsubscribe`, and `handle_event` methods don't exist.

**Step 3: Implement subscribe and unsubscribe**

Add these methods to the `impl LivelinessRouter` block. Also add the `use zenoh_keyexpr::key_expr::keyexpr;` import at the top of the file.

```rust
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

    // History: emit TokenAppeared for each existing remote token that matches.
    let mut actions = Vec::new();
    for (_table_id, token_key) in &self.remote_token_keys {
        let token_ke = keyexpr::new(token_key)
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        if owned.intersects(token_ke) {
            actions.push(LivelinessAction::TokenAppeared {
                subscriber_id: sub_id,
                key_expr: token_key.clone(),
            });
        }
    }

    Ok((sub_id, actions))
}

/// Unsubscribe a liveliness subscriber.
pub fn unsubscribe(
    &mut self,
    subscriber_id: LivelinessSubscriberId,
) -> Result<(), ZenohError> {
    let sub_table_id = self
        .subscriber_keys
        .remove(&subscriber_id)
        .ok_or(ZenohError::UnknownSubscriptionId(subscriber_id))?;
    self.subscribers.unsubscribe(sub_table_id)?;
    Ok(())
}
```

**Step 4: Run tests to verify they pass**

These tests also depend on `handle_event`, which doesn't exist yet. The `subscribe_returns_id` and `subscribe_rejects_invalid_key_expr` and `unsubscribe_unknown_id_fails` tests will pass. The tests calling `handle_event` will fail — that's expected, we implement `handle_event` in Task 5.

Run: `cargo test -p harmony-zenoh liveliness::tests::subscribe_returns_id liveliness::tests::subscribe_rejects_invalid_key_expr liveliness::tests::unsubscribe_unknown_id_fails`
Expected: 3 tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/liveliness.rs
git commit -m "feat(zenoh): implement liveliness subscribe and unsubscribe"
```

---

### Task 5: Implement handle_event (TokenDeclared, TokenUndeclared, PeerLost)

**Files:**
- Modify: `crates/harmony-zenoh/src/liveliness.rs`

**Step 1: Write the failing tests**

Add to the `tests` module:

```rust
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
    // Undeclare a token that was never declared — should be a no-op
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
    let (sub_id, _) = router.subscribe("harmony/presence/srv1/*").unwrap();
    // First declare
    let actions1 = router
        .handle_event(LivelinessEvent::TokenDeclared {
            key_expr: "harmony/presence/srv1/alice".into(),
        })
        .unwrap();
    assert_eq!(actions1.len(), 1);
    // Second declare on same key — idempotent, no spurious events
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
    let (sub_id, _) = router.subscribe("harmony/presence/**").unwrap();
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

    let actions = router
        .handle_event(LivelinessEvent::PeerLost)
        .unwrap();

    // Should get TokenDisappeared for both tokens
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
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh liveliness`
Expected: FAIL — `handle_event` method doesn't exist.

**Step 3: Implement handle_event**

Add to `impl LivelinessRouter`:

```rust
/// Process an inbound liveliness event and return actions for the caller.
pub fn handle_event(
    &mut self,
    event: LivelinessEvent,
) -> Result<Vec<LivelinessAction>, ZenohError> {
    match event {
        LivelinessEvent::TokenDeclared { key_expr } => {
            self.handle_token_declared(key_expr)
        }
        LivelinessEvent::TokenUndeclared { key_expr } => {
            self.handle_token_undeclared(key_expr)
        }
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
    if self.remote_token_keys.values().any(|k| k == &canonical) {
        return Ok(vec![]);
    }

    let sub_id = self.remote_tokens.subscribe(&owned);
    self.remote_token_keys.insert(sub_id, canonical.clone());

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
    let table_id = self
        .remote_token_keys
        .iter()
        .find(|(_, k)| *k == &canonical)
        .map(|(id, _)| *id);

    let Some(table_id) = table_id else {
        return Ok(vec![]); // Unknown token — no-op.
    };

    self.remote_token_keys.remove(&table_id);
    let _ = self.remote_tokens.unsubscribe(table_id);

    self.notify_subscribers_disappeared(&canonical)
}

fn handle_peer_lost(&mut self) -> Vec<LivelinessAction> {
    let token_keys: Vec<(SubscriptionId, String)> =
        self.remote_token_keys.drain().collect();
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
    let ke = keyexpr::new(key_expr)
        .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
    let matches = self.subscribers.matches(ke);
    let mut actions = Vec::new();
    for sub_table_id in matches {
        // Reverse lookup: find the LivelinessSubscriberId for this table ID.
        if let Some((&sub_id, _)) = self
            .subscriber_keys
            .iter()
            .find(|(_, &table_id)| table_id == sub_table_id)
        {
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
    let ke = keyexpr::new(key_expr)
        .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
    let matches = self.subscribers.matches(ke);
    let mut actions = Vec::new();
    for sub_table_id in matches {
        if let Some((&sub_id, _)) = self
            .subscriber_keys
            .iter()
            .find(|(_, &table_id)| table_id == sub_table_id)
        {
            actions.push(LivelinessAction::TokenDisappeared {
                subscriber_id: sub_id,
                key_expr: key_expr.to_string(),
            });
        }
    }
    Ok(actions)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-zenoh liveliness`
Expected: All tests pass (including Task 4's tests that depend on handle_event).

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/liveliness.rs
git commit -m "feat(zenoh): implement handle_event for liveliness tokens"
```

---

### Task 6: Implement query

**Files:**
- Modify: `crates/harmony-zenoh/src/liveliness.rs`

**Step 1: Write the failing tests**

Add to the `tests` module:

```rust
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
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh liveliness::tests::query`
Expected: FAIL — `query` method doesn't exist.

**Step 3: Implement query**

Add to `impl LivelinessRouter`:

```rust
/// Query currently alive remote tokens matching the given key expression.
///
/// Returns the key expressions of all matching tokens.
pub fn query(&self, key_expr: &str) -> Result<Vec<String>, ZenohError> {
    let owned = OwnedKeyExpr::autocanonize(key_expr.to_string())
        .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
    let mut results = Vec::new();
    for token_key in self.remote_token_keys.values() {
        let token_ke = keyexpr::new(token_key)
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        if owned.intersects(token_ke) {
            results.push(token_key.clone());
        }
    }
    Ok(results)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-zenoh liveliness`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/liveliness.rs
git commit -m "feat(zenoh): implement liveliness query"
```

---

### Task 7: Run full quality gates

**Files:** None (verification only).

**Step 1: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass (409+ tests).

**Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: No warnings.

**Step 3: Run format check**

Run: `cargo fmt --all -- --check`
Expected: Clean (or only pre-existing drift).

**Step 4: Run harmony-zenoh tests in isolation**

Run: `cargo test -p harmony-zenoh`
Expected: All tests pass.

**Step 5: Commit any clippy/fmt fixes if needed**

```bash
# Only if needed:
cargo fmt --all
git add -u
git commit -m "style: format liveliness module"
```

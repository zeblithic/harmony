# Session State Machine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a sans-I/O session state machine to harmony-zenoh that manages peer-to-peer connections with identity-based handshake, resource ID mapping, keepalive, and graceful close.

**Architecture:** A `Session` struct driven by `handle_event(SessionEvent) → Vec<SessionAction>`, mirroring harmony-reticulum's `Node` pattern. Each session represents one two-party encrypted connection. The session does not own SubscriptionTable or HarmonyEnvelope — the caller composes those.

**Tech Stack:** Rust, harmony-identity (Ed25519 sign/verify), std HashMap, no new external deps.

---

### Task 1: Add error variants and session module skeleton

Add the 4 new error variants to `ZenohError`, create `session.rs` with the core types (enums, config struct, ExprId type alias), and wire up `lib.rs`. No logic yet — just types that compile.

**Files:**
- Modify: `crates/harmony-zenoh/src/error.rs:3-24`
- Create: `crates/harmony-zenoh/src/session.rs`
- Modify: `crates/harmony-zenoh/src/lib.rs:1-9`

**Step 1: Add error variants to error.rs**

Add these 4 variants inside the `ZenohError` enum (after the existing `OpenFailed` variant at line 23):

```rust
    #[error("session not active")]
    SessionNotActive,

    #[error("handshake failed: {0}")]
    HandshakeFailed(String),

    #[error("duplicate expression ID: {0}")]
    DuplicateExprId(u64),

    #[error("unknown expression ID: {0}")]
    UnknownExprId(u64),
```

**Step 2: Create session.rs with core types**

```rust
//! Sans-I/O session state machine for peer-to-peer connections.
//!
//! Each `Session` manages a single two-party encrypted connection:
//! identity-based handshake, resource ID mapping, keepalive, and
//! graceful close. The caller drives the session via events and
//! executes the returned actions.

use std::collections::HashMap;

use harmony_identity::{Identity, PrivateIdentity};

use crate::ZenohError;

/// Numeric expression ID for wire-efficient key expression references.
pub type ExprId = u64;

/// Session lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Awaiting mutual identity handshake.
    Init,
    /// Authenticated, resources can be declared/undeclared.
    Active,
    /// Close message sent, waiting for ack.
    Closing,
    /// Terminal state, no further events accepted.
    Closed,
}

/// Tunable session timing parameters.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Interval between keepalive emissions (ms).
    pub keepalive_interval_ms: u64,
    /// Time without any inbound event before declaring peer stale (ms).
    pub stale_timeout_ms: u64,
    /// Time to wait for CloseAck before force-closing (ms).
    pub close_timeout_ms: u64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            keepalive_interval_ms: 30_000,
            stale_timeout_ms: 90_000,
            close_timeout_ms: 5_000,
        }
    }
}

/// Inbound events the caller feeds into the session.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// Peer's handshake proof received.
    HandshakeReceived { proof: Vec<u8> },
    /// Peer declared a resource (key expression ↔ ExprId mapping).
    ResourceDeclared { expr_id: ExprId, key_expr: String },
    /// Peer undeclared a resource.
    ResourceUndeclared { expr_id: ExprId },
    /// Peer sent a keepalive.
    KeepaliveReceived,
    /// Peer initiated close.
    CloseReceived,
    /// Peer acknowledged our close.
    CloseAckReceived,
    /// Timer tick with caller-provided timestamp.
    TimerTick { now_ms: u64 },
}

/// Outbound actions the session returns for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionAction {
    /// Send our handshake proof to the peer.
    SendHandshake { proof: Vec<u8> },
    /// Send a keepalive to the peer.
    SendKeepalive,
    /// Send a close request to the peer.
    SendClose,
    /// Send a close acknowledgment to the peer.
    SendCloseAck,
    /// Send a resource declaration to the peer.
    SendResourceDeclare { expr_id: ExprId, key_expr: String },
    /// Send a resource undeclaration to the peer.
    SendResourceUndeclare { expr_id: ExprId },
    /// A new remote resource is available.
    ResourceAdded { expr_id: ExprId, key_expr: String },
    /// A remote resource was removed.
    ResourceRemoved { expr_id: ExprId },
    /// Peer has gone stale (no inbound events within timeout).
    PeerStale,
    /// Session is now authenticated and active.
    SessionOpened,
    /// Session has been closed (gracefully or forced).
    SessionClosed,
}

/// A sans-I/O session managing a single peer-to-peer connection.
pub struct Session {
    state: SessionState,
    local_identity: PrivateIdentity,
    remote_identity: Identity,
    handshake_sent: bool,
    handshake_verified: bool,
    local_resources: HashMap<ExprId, String>,
    remote_resources: HashMap<ExprId, String>,
    next_expr_id: ExprId,
    last_received_ms: u64,
    last_sent_ms: u64,
    last_tick_ms: u64,
    close_initiated_ms: Option<u64>,
    config: SessionConfig,
}
```

**Step 3: Wire up lib.rs**

Replace `lib.rs` contents with:

```rust
pub mod envelope;
pub mod error;
pub mod keyspace;
pub mod session;
pub mod subscription;

pub use envelope::{HarmonyEnvelope, MessageType, HEADER_SIZE, MIN_ENVELOPE_SIZE, VERSION};
pub use error::ZenohError;
pub use keyspace::keyexpr;
pub use session::{ExprId, Session, SessionAction, SessionConfig, SessionEvent, SessionState};
pub use subscription::{SubscriptionId, SubscriptionTable};
```

**Step 4: Verify it compiles**

Run: `cargo check -p harmony-zenoh`
Expected: compiles with no errors (may have dead_code warnings, that's fine)

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/error.rs crates/harmony-zenoh/src/session.rs crates/harmony-zenoh/src/lib.rs
git commit -m "Add session module skeleton with core types and error variants"
```

---

### Task 2: Handshake — new(), handle_event dispatch, and Init → Active

Implement `Session::new()` (which emits `SendHandshake`), the `handle_event()` dispatch method, and the handshake verification logic. This is the first real behavior.

**Files:**
- Modify: `crates/harmony-zenoh/src/session.rs`

**Reference docs:**
- `crates/harmony-identity/src/identity.rs:180` — `PrivateIdentity::sign(message) -> [u8; 64]`
- `crates/harmony-identity/src/identity.rs:87` — `Identity::verify(message, signature) -> Result<(), IdentityError>`
- `crates/harmony-identity/src/identity.rs:218` — `PrivateIdentity::public_identity() -> &Identity`
- `crates/harmony-identity/src/identity.rs:43` — `Identity::address_hash: [u8; 16]`

**Important:** `PrivateIdentity` contains `StaticSecret` which does NOT implement `Clone`. The `Session::new()` takes ownership (move). Test helpers must extract public identities before moving.

**Step 1: Write the failing tests**

Add a `#[cfg(test)]` module at the bottom of `session.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    /// Helper: create a session pair (alice's session, bob's session) and
    /// the initial SendHandshake actions from each.
    fn create_session_pair() -> (Session, Vec<SessionAction>, Session, Vec<SessionAction>) {
        let mut rng = OsRng;
        let alice_id = PrivateIdentity::generate(&mut rng);
        let bob_id = PrivateIdentity::generate(&mut rng);

        let alice_pub = alice_id.public_identity().clone();
        let bob_pub = bob_id.public_identity().clone();

        let (alice_session, alice_actions) = Session::new(
            alice_id,
            bob_pub,
            SessionConfig::default(),
            0,
        );
        let (bob_session, bob_actions) = Session::new(
            bob_id,
            alice_pub,
            SessionConfig::default(),
            0,
        );

        (alice_session, alice_actions, bob_session, bob_actions)
    }

    /// Helper: extract the handshake proof from SendHandshake action.
    fn extract_handshake_proof(actions: &[SessionAction]) -> Vec<u8> {
        actions
            .iter()
            .find_map(|a| match a {
                SessionAction::SendHandshake { proof } => Some(proof.clone()),
                _ => None,
            })
            .expect("expected SendHandshake action")
    }

    /// Helper: complete the handshake for both sides, returning both to Active.
    fn complete_handshake(
        alice: &mut Session,
        alice_actions: &[SessionAction],
        bob: &mut Session,
        bob_actions: &[SessionAction],
    ) {
        let alice_proof = extract_handshake_proof(alice_actions);
        let bob_proof = extract_handshake_proof(bob_actions);

        let bob_result = bob
            .handle_event(SessionEvent::HandshakeReceived { proof: alice_proof })
            .unwrap();
        assert!(bob_result.contains(&SessionAction::SessionOpened));
        assert_eq!(bob.state(), SessionState::Active);

        let alice_result = alice
            .handle_event(SessionEvent::HandshakeReceived { proof: bob_proof })
            .unwrap();
        assert!(alice_result.contains(&SessionAction::SessionOpened));
        assert_eq!(alice.state(), SessionState::Active);
    }

    #[test]
    fn handshake_roundtrip() {
        let (mut alice, alice_actions, mut bob, bob_actions) = create_session_pair();

        assert_eq!(alice.state(), SessionState::Init);
        assert_eq!(bob.state(), SessionState::Init);
        assert_eq!(alice_actions.len(), 1);
        assert!(matches!(&alice_actions[0], SessionAction::SendHandshake { .. }));

        complete_handshake(&mut alice, &alice_actions, &mut bob, &bob_actions);
    }

    #[test]
    fn handshake_wrong_identity_fails() {
        let mut rng = OsRng;
        let alice_id = PrivateIdentity::generate(&mut rng);
        let bob_id = PrivateIdentity::generate(&mut rng);
        let eve_id = PrivateIdentity::generate(&mut rng);

        let bob_pub = bob_id.public_identity().clone();
        let alice_pub = alice_id.public_identity().clone();

        let (mut bob_session, _) = Session::new(
            bob_id,
            alice_pub,
            SessionConfig::default(),
            0,
        );

        // Eve signs a proof, but Bob expects Alice
        let eve_proof = eve_id.sign(
            &[b"harmony-session-v1" as &[u8], &bob_pub.address_hash].concat(),
        );

        let result = bob_session.handle_event(SessionEvent::HandshakeReceived {
            proof: eve_proof.to_vec(),
        });
        assert!(result.is_err());
        assert_eq!(bob_session.state(), SessionState::Closed);
    }

    #[test]
    fn handshake_replay_rejected() {
        let mut rng = OsRng;
        let alice_id = PrivateIdentity::generate(&mut rng);
        let bob_id = PrivateIdentity::generate(&mut rng);
        let charlie_id = PrivateIdentity::generate(&mut rng);

        let alice_pub = alice_id.public_identity().clone();
        let bob_pub = bob_id.public_identity().clone();

        // Alice signs proof for Bob
        let proof_for_bob = alice_id.sign(
            &[b"harmony-session-v1" as &[u8], &bob_pub.address_hash].concat(),
        );

        // Try to replay that proof to Charlie (who expects Alice)
        let (mut charlie_session, _) = Session::new(
            charlie_id,
            alice_pub,
            SessionConfig::default(),
            0,
        );

        // Charlie's address_hash != Bob's, so verification should fail
        let result = charlie_session.handle_event(SessionEvent::HandshakeReceived {
            proof: proof_for_bob.to_vec(),
        });
        assert!(result.is_err());
        assert_eq!(charlie_session.state(), SessionState::Closed);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh session::tests --no-run 2>&1 | head -20`
Expected: FAIL — `Session::new` and `handle_event` don't exist yet.

**Step 3: Implement Session::new, state(), and handle_event with handshake logic**

Add to `session.rs` above the test module:

```rust
/// Handshake info string prefix.
const HANDSHAKE_INFO: &[u8] = b"harmony-session-v1";

impl Session {
    /// Create a new session and produce the initial handshake action.
    ///
    /// Returns the session and the initial actions (always a single
    /// `SendHandshake` containing the Ed25519 proof).
    pub fn new(
        local_identity: PrivateIdentity,
        remote_identity: Identity,
        config: SessionConfig,
        now_ms: u64,
    ) -> (Self, Vec<SessionAction>) {
        // Sign: "harmony-session-v1" || remote_address_hash
        let mut msg = Vec::with_capacity(HANDSHAKE_INFO.len() + 16);
        msg.extend_from_slice(HANDSHAKE_INFO);
        msg.extend_from_slice(&remote_identity.address_hash);
        let proof = local_identity.sign(&msg);

        let session = Self {
            state: SessionState::Init,
            local_identity,
            remote_identity,
            handshake_sent: true,
            handshake_verified: false,
            local_resources: HashMap::new(),
            remote_resources: HashMap::new(),
            next_expr_id: 1,
            last_received_ms: now_ms,
            last_sent_ms: now_ms,
            last_tick_ms: now_ms,
            close_initiated_ms: None,
            config,
        };

        let actions = vec![SessionAction::SendHandshake {
            proof: proof.to_vec(),
        }];

        (session, actions)
    }

    /// Current lifecycle state.
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Process an inbound event and return actions for the caller to execute.
    pub fn handle_event(
        &mut self,
        event: SessionEvent,
    ) -> Result<Vec<SessionAction>, ZenohError> {
        match event {
            SessionEvent::HandshakeReceived { proof } => self.handle_handshake(proof),
            SessionEvent::TimerTick { now_ms } => self.handle_timer_tick(now_ms),
            _ => {
                if self.state == SessionState::Closed || self.state == SessionState::Init {
                    return Err(ZenohError::SessionNotActive);
                }
                match event {
                    SessionEvent::ResourceDeclared { expr_id, key_expr } => {
                        self.handle_resource_declared(expr_id, key_expr)
                    }
                    SessionEvent::ResourceUndeclared { expr_id } => {
                        self.handle_resource_undeclared(expr_id)
                    }
                    SessionEvent::KeepaliveReceived => self.handle_keepalive_received(),
                    SessionEvent::CloseReceived => self.handle_close_received(),
                    SessionEvent::CloseAckReceived => self.handle_close_ack_received(),
                    SessionEvent::HandshakeReceived { .. } | SessionEvent::TimerTick { .. } => {
                        unreachable!()
                    }
                }
            }
        }
    }

    fn handle_handshake(&mut self, proof: Vec<u8>) -> Result<Vec<SessionAction>, ZenohError> {
        if self.state != SessionState::Init {
            return Err(ZenohError::SessionNotActive);
        }

        // Proof must be exactly 64 bytes (Ed25519 signature)
        let signature: [u8; 64] = proof.try_into().map_err(|_| {
            ZenohError::HandshakeFailed("invalid proof length".into())
        })?;

        // Verify: remote signed "harmony-session-v1" || our_address_hash
        let mut msg = Vec::with_capacity(HANDSHAKE_INFO.len() + 16);
        msg.extend_from_slice(HANDSHAKE_INFO);
        msg.extend_from_slice(&self.local_identity.public_identity().address_hash);

        if self.remote_identity.verify(&msg, &signature).is_err() {
            self.state = SessionState::Closed;
            return Err(ZenohError::HandshakeFailed(
                "signature verification failed".into(),
            ));
        }

        self.handshake_verified = true;
        self.last_received_ms = self.last_tick_ms;

        if self.handshake_sent && self.handshake_verified {
            self.state = SessionState::Active;
            Ok(vec![SessionAction::SessionOpened])
        } else {
            Ok(vec![])
        }
    }

    // Stub methods — implemented in later tasks
    fn handle_timer_tick(&mut self, _now_ms: u64) -> Result<Vec<SessionAction>, ZenohError> {
        Ok(vec![])
    }

    fn handle_resource_declared(
        &mut self,
        _expr_id: ExprId,
        _key_expr: String,
    ) -> Result<Vec<SessionAction>, ZenohError> {
        Ok(vec![])
    }

    fn handle_resource_undeclared(
        &mut self,
        _expr_id: ExprId,
    ) -> Result<Vec<SessionAction>, ZenohError> {
        Ok(vec![])
    }

    fn handle_keepalive_received(&mut self) -> Result<Vec<SessionAction>, ZenohError> {
        Ok(vec![])
    }

    fn handle_close_received(&mut self) -> Result<Vec<SessionAction>, ZenohError> {
        Ok(vec![])
    }

    fn handle_close_ack_received(&mut self) -> Result<Vec<SessionAction>, ZenohError> {
        Ok(vec![])
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-zenoh session::tests -- --nocapture`
Expected: 3 tests pass (handshake_roundtrip, handshake_wrong_identity_fails, handshake_replay_rejected)

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/session.rs
git commit -m "Implement session handshake with identity-based authentication"
```

---

### Task 3: Resource declaration and undeclaration

Implement the resource mapping logic: `declare_resource()` for local resources, `handle_resource_declared()` / `handle_resource_undeclared()` for remote resources, and `resolve_remote()` / `resolve_local()` lookup methods.

**Files:**
- Modify: `crates/harmony-zenoh/src/session.rs`

**Step 1: Write the failing tests**

Add to the existing `tests` module:

```rust
    #[test]
    fn resource_declare_undeclare_lifecycle() {
        let (mut alice, alice_actions, mut bob, bob_actions) = create_session_pair();
        complete_handshake(&mut alice, &alice_actions, &mut bob, &bob_actions);

        // Alice declares a local resource
        let (expr_id, actions) = alice.declare_resource("harmony/server/srv1/channel/general/msg".into());
        assert_eq!(expr_id, 1);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            SessionAction::SendResourceDeclare { expr_id: 1, key_expr } if key_expr == "harmony/server/srv1/channel/general/msg"
        ));

        // Bob receives the declaration
        let actions = bob
            .handle_event(SessionEvent::ResourceDeclared {
                expr_id: 1,
                key_expr: "harmony/server/srv1/channel/general/msg".into(),
            })
            .unwrap();
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            SessionAction::ResourceAdded { expr_id: 1, key_expr } if key_expr == "harmony/server/srv1/channel/general/msg"
        ));

        // Bob can resolve the remote resource
        assert_eq!(
            bob.resolve_remote(1),
            Some("harmony/server/srv1/channel/general/msg")
        );

        // Bob receives undeclare
        let actions = bob
            .handle_event(SessionEvent::ResourceUndeclared { expr_id: 1 })
            .unwrap();
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::ResourceRemoved { expr_id: 1 }));
        assert_eq!(bob.resolve_remote(1), None);
    }

    #[test]
    fn duplicate_expr_id_rejected() {
        let (mut alice, alice_actions, mut bob, bob_actions) = create_session_pair();
        complete_handshake(&mut alice, &alice_actions, &mut bob, &bob_actions);

        bob.handle_event(SessionEvent::ResourceDeclared {
            expr_id: 5,
            key_expr: "harmony/dm/abc/msg".into(),
        })
        .unwrap();

        let result = bob.handle_event(SessionEvent::ResourceDeclared {
            expr_id: 5,
            key_expr: "harmony/dm/xyz/msg".into(),
        });
        assert!(matches!(result, Err(ZenohError::DuplicateExprId(5))));
    }

    #[test]
    fn unknown_expr_id_undeclare_rejected() {
        let (mut alice, alice_actions, mut bob, bob_actions) = create_session_pair();
        complete_handshake(&mut alice, &alice_actions, &mut bob, &bob_actions);

        let result = bob.handle_event(SessionEvent::ResourceUndeclared { expr_id: 99 });
        assert!(matches!(result, Err(ZenohError::UnknownExprId(99))));
    }

    #[test]
    fn local_expr_id_monotonic() {
        let (mut alice, alice_actions, mut bob, bob_actions) = create_session_pair();
        complete_handshake(&mut alice, &alice_actions, &mut bob, &bob_actions);

        let (id1, _) = alice.declare_resource("harmony/server/a/msg".into());
        let (id2, _) = alice.declare_resource("harmony/server/b/msg".into());
        let (id3, _) = alice.declare_resource("harmony/server/c/msg".into());

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh session::tests --no-run 2>&1 | head -20`
Expected: FAIL — `declare_resource`, `resolve_remote` don't exist.

**Step 3: Implement resource management**

Add these methods to `impl Session` (replace the resource handler stubs):

```rust
    /// Declare a local resource, allocating the next ExprId.
    ///
    /// Returns the assigned ExprId and a `SendResourceDeclare` action for the
    /// caller to transmit to the peer.
    pub fn declare_resource(&mut self, key_expr: String) -> (ExprId, Vec<SessionAction>) {
        let expr_id = self.next_expr_id;
        self.next_expr_id += 1;
        self.local_resources.insert(expr_id, key_expr.clone());

        let actions = vec![SessionAction::SendResourceDeclare { expr_id, key_expr }];
        (expr_id, actions)
    }

    /// Undeclare a local resource.
    ///
    /// Returns a `SendResourceUndeclare` action for the caller to transmit.
    pub fn undeclare_resource(&mut self, expr_id: ExprId) -> Result<Vec<SessionAction>, ZenohError> {
        if self.local_resources.remove(&expr_id).is_none() {
            return Err(ZenohError::UnknownExprId(expr_id));
        }
        Ok(vec![SessionAction::SendResourceUndeclare { expr_id }])
    }

    /// Look up a remote resource's key expression by ExprId.
    pub fn resolve_remote(&self, expr_id: ExprId) -> Option<&str> {
        self.remote_resources.get(&expr_id).map(|s| s.as_str())
    }

    /// Look up a local resource's key expression by ExprId.
    pub fn resolve_local(&self, expr_id: ExprId) -> Option<&str> {
        self.local_resources.get(&expr_id).map(|s| s.as_str())
    }

    fn handle_resource_declared(
        &mut self,
        expr_id: ExprId,
        key_expr: String,
    ) -> Result<Vec<SessionAction>, ZenohError> {
        self.last_received_ms = self.last_tick_ms;
        if self.remote_resources.contains_key(&expr_id) {
            return Err(ZenohError::DuplicateExprId(expr_id));
        }
        self.remote_resources.insert(expr_id, key_expr.clone());
        Ok(vec![SessionAction::ResourceAdded { expr_id, key_expr }])
    }

    fn handle_resource_undeclared(
        &mut self,
        expr_id: ExprId,
    ) -> Result<Vec<SessionAction>, ZenohError> {
        self.last_received_ms = self.last_tick_ms;
        if self.remote_resources.remove(&expr_id).is_none() {
            return Err(ZenohError::UnknownExprId(expr_id));
        }
        Ok(vec![SessionAction::ResourceRemoved { expr_id }])
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-zenoh session::tests -- --nocapture`
Expected: 7 tests pass (3 handshake + 4 resource)

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/session.rs
git commit -m "Implement resource declaration, undeclaration, and resolution"
```

---

### Task 4: Keepalive and stale peer detection

Implement the timer tick handler: emit `SendKeepalive` after interval, detect stale peers, and update timestamps on inbound events.

**Files:**
- Modify: `crates/harmony-zenoh/src/session.rs`

**Step 1: Write the failing tests**

Add to the existing `tests` module:

```rust
    #[test]
    fn keepalive_emitted_after_interval() {
        let config = SessionConfig {
            keepalive_interval_ms: 100,
            stale_timeout_ms: 300,
            close_timeout_ms: 50,
        };
        let mut rng = OsRng;
        let alice_id = PrivateIdentity::generate(&mut rng);
        let bob_id = PrivateIdentity::generate(&mut rng);
        let alice_pub = alice_id.public_identity().clone();
        let bob_pub = bob_id.public_identity().clone();

        let (mut alice, alice_actions) = Session::new(alice_id, bob_pub, config.clone(), 0);
        let (mut bob, bob_actions) = Session::new(bob_id, alice_pub, config, 0);
        complete_handshake(&mut alice, &alice_actions, &mut bob, &bob_actions);

        // Before interval: no keepalive
        let actions = alice.handle_event(SessionEvent::TimerTick { now_ms: 50 }).unwrap();
        assert!(!actions.contains(&SessionAction::SendKeepalive));

        // At interval: keepalive emitted
        let actions = alice.handle_event(SessionEvent::TimerTick { now_ms: 100 }).unwrap();
        assert!(actions.contains(&SessionAction::SendKeepalive));
    }

    #[test]
    fn stale_peer_detected_after_timeout() {
        let config = SessionConfig {
            keepalive_interval_ms: 100,
            stale_timeout_ms: 300,
            close_timeout_ms: 50,
        };
        let mut rng = OsRng;
        let alice_id = PrivateIdentity::generate(&mut rng);
        let bob_id = PrivateIdentity::generate(&mut rng);
        let alice_pub = alice_id.public_identity().clone();
        let bob_pub = bob_id.public_identity().clone();

        let (mut alice, alice_actions) = Session::new(alice_id, bob_pub, config.clone(), 0);
        let (mut bob, bob_actions) = Session::new(bob_id, alice_pub, config, 0);
        complete_handshake(&mut alice, &alice_actions, &mut bob, &bob_actions);

        // Not yet stale
        let actions = alice.handle_event(SessionEvent::TimerTick { now_ms: 200 }).unwrap();
        assert!(!actions.contains(&SessionAction::PeerStale));
        assert_eq!(alice.state(), SessionState::Active);

        // Stale
        let actions = alice.handle_event(SessionEvent::TimerTick { now_ms: 300 }).unwrap();
        assert!(actions.contains(&SessionAction::PeerStale));
        assert!(actions.contains(&SessionAction::SessionClosed));
        assert_eq!(alice.state(), SessionState::Closed);
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh session::tests::keepalive -- --nocapture`
Expected: FAIL — timer tick stub returns empty actions.

**Step 3: Implement timer tick and timestamp tracking**

Replace the `handle_timer_tick` stub and add the `force_close` helper. Also replace the `handle_keepalive_received` stub:

```rust
    fn handle_timer_tick(&mut self, now_ms: u64) -> Result<Vec<SessionAction>, ZenohError> {
        let mut actions = Vec::new();
        self.last_tick_ms = now_ms;

        if self.state == SessionState::Closed {
            return Ok(actions);
        }

        // Check stale timeout (applies in Init, Active, and Closing)
        if now_ms.saturating_sub(self.last_received_ms) >= self.config.stale_timeout_ms {
            actions.extend(self.force_close());
            actions.push(SessionAction::PeerStale);
            return Ok(actions);
        }

        // Check close timeout
        if self.state == SessionState::Closing {
            if let Some(close_ms) = self.close_initiated_ms {
                if now_ms.saturating_sub(close_ms) >= self.config.close_timeout_ms {
                    actions.extend(self.force_close());
                    return Ok(actions);
                }
            }
        }

        // Emit keepalive if interval elapsed (only in Active state)
        if self.state == SessionState::Active
            && now_ms.saturating_sub(self.last_sent_ms) >= self.config.keepalive_interval_ms
        {
            self.last_sent_ms = now_ms;
            actions.push(SessionAction::SendKeepalive);
        }

        Ok(actions)
    }

    /// Force-close the session, bulk-undeclaring all resources.
    fn force_close(&mut self) -> Vec<SessionAction> {
        self.state = SessionState::Closed;
        let mut actions: Vec<SessionAction> = self
            .remote_resources
            .keys()
            .copied()
            .collect::<Vec<_>>()
            .into_iter()
            .map(|expr_id| SessionAction::ResourceRemoved { expr_id })
            .collect();
        self.remote_resources.clear();
        self.local_resources.clear();
        actions.push(SessionAction::SessionClosed);
        actions
    }

    fn handle_keepalive_received(&mut self) -> Result<Vec<SessionAction>, ZenohError> {
        self.last_received_ms = self.last_tick_ms;
        Ok(vec![])
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-zenoh session::tests -- --nocapture`
Expected: 9 tests pass (3 handshake + 4 resource + 2 keepalive/stale)

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/session.rs
git commit -m "Implement keepalive emission and stale peer detection"
```

---

### Task 5: Close handshake (graceful and remote-initiated)

Implement `initiate_close()`, the close/ack handlers, close timeout in timer tick (already implemented), and remote-initiated close.

**Files:**
- Modify: `crates/harmony-zenoh/src/session.rs`

**Step 1: Write the failing tests**

Add to the existing `tests` module:

```rust
    #[test]
    fn graceful_close_handshake() {
        let config = SessionConfig {
            keepalive_interval_ms: 100,
            stale_timeout_ms: 300,
            close_timeout_ms: 50,
        };
        let mut rng = OsRng;
        let alice_id = PrivateIdentity::generate(&mut rng);
        let bob_id = PrivateIdentity::generate(&mut rng);
        let alice_pub = alice_id.public_identity().clone();
        let bob_pub = bob_id.public_identity().clone();

        let (mut alice, alice_actions) = Session::new(alice_id, bob_pub, config.clone(), 0);
        let (mut bob, bob_actions) = Session::new(bob_id, alice_pub, config, 0);
        complete_handshake(&mut alice, &alice_actions, &mut bob, &bob_actions);

        // Alice initiates close
        let actions = alice.initiate_close(10).unwrap();
        assert!(actions.contains(&SessionAction::SendClose));
        assert_eq!(alice.state(), SessionState::Closing);

        // Alice receives close ack
        let actions = alice
            .handle_event(SessionEvent::CloseAckReceived)
            .unwrap();
        assert!(actions.contains(&SessionAction::SessionClosed));
        assert_eq!(alice.state(), SessionState::Closed);
    }

    #[test]
    fn close_timeout_forces_closed() {
        let config = SessionConfig {
            keepalive_interval_ms: 100,
            stale_timeout_ms: 300,
            close_timeout_ms: 50,
        };
        let mut rng = OsRng;
        let alice_id = PrivateIdentity::generate(&mut rng);
        let bob_id = PrivateIdentity::generate(&mut rng);
        let alice_pub = alice_id.public_identity().clone();
        let bob_pub = bob_id.public_identity().clone();

        let (mut alice, alice_actions) = Session::new(alice_id, bob_pub, config.clone(), 0);
        let (mut bob, bob_actions) = Session::new(bob_id, alice_pub, config, 0);
        complete_handshake(&mut alice, &alice_actions, &mut bob, &bob_actions);

        alice.initiate_close(10).unwrap();

        // Not yet timed out
        let actions = alice.handle_event(SessionEvent::TimerTick { now_ms: 30 }).unwrap();
        assert!(!actions.contains(&SessionAction::SessionClosed));

        // Timed out — force close
        let actions = alice.handle_event(SessionEvent::TimerTick { now_ms: 60 }).unwrap();
        assert!(actions.contains(&SessionAction::SessionClosed));
        assert_eq!(alice.state(), SessionState::Closed);
    }

    #[test]
    fn remote_initiated_close() {
        let config = SessionConfig {
            keepalive_interval_ms: 100,
            stale_timeout_ms: 300,
            close_timeout_ms: 50,
        };
        let mut rng = OsRng;
        let alice_id = PrivateIdentity::generate(&mut rng);
        let bob_id = PrivateIdentity::generate(&mut rng);
        let alice_pub = alice_id.public_identity().clone();
        let bob_pub = bob_id.public_identity().clone();

        let (mut alice, alice_actions) = Session::new(alice_id, bob_pub, config.clone(), 0);
        let (mut bob, bob_actions) = Session::new(bob_id, alice_pub, config, 0);
        complete_handshake(&mut alice, &alice_actions, &mut bob, &bob_actions);

        // Bob receives close from Alice
        let actions = bob
            .handle_event(SessionEvent::CloseReceived)
            .unwrap();
        assert!(actions.contains(&SessionAction::SendCloseAck));
        assert!(actions.contains(&SessionAction::SessionClosed));
        assert_eq!(bob.state(), SessionState::Closed);
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh session::tests::close -- --no-run`
Expected: FAIL — `initiate_close` doesn't exist, close handlers are stubs.

**Step 3: Implement close logic**

Add `initiate_close` and replace the close handler stubs:

```rust
    /// Initiate a graceful close. Transitions to Closing and emits `SendClose`.
    pub fn initiate_close(&mut self, now_ms: u64) -> Result<Vec<SessionAction>, ZenohError> {
        if self.state != SessionState::Active {
            return Err(ZenohError::SessionNotActive);
        }
        self.state = SessionState::Closing;
        self.close_initiated_ms = Some(now_ms);
        Ok(vec![SessionAction::SendClose])
    }

    fn handle_close_received(&mut self) -> Result<Vec<SessionAction>, ZenohError> {
        self.last_received_ms = self.last_tick_ms;
        let mut actions = vec![SessionAction::SendCloseAck];
        actions.extend(self.force_close());
        Ok(actions)
    }

    fn handle_close_ack_received(&mut self) -> Result<Vec<SessionAction>, ZenohError> {
        self.last_received_ms = self.last_tick_ms;
        Ok(self.force_close())
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-zenoh session::tests -- --nocapture`
Expected: 12 tests pass (3 handshake + 4 resource + 2 keepalive + 3 close)

**Step 5: Run full workspace verification**

Run: `cargo test --workspace`
Expected: All tests pass (379+ tests)

Run: `cargo clippy --workspace`
Expected: No warnings

**Step 6: Commit**

```bash
git add crates/harmony-zenoh/src/session.rs
git commit -m "Implement graceful close handshake with timeout fallback"
```

---

### Summary

| Task | What | Tests |
|------|------|-------|
| 1 | Error variants + types skeleton | 0 (compile check) |
| 2 | Handshake (new, handle_event, Init→Active) | 3 |
| 3 | Resource declare/undeclare/resolve | 4 |
| 4 | Keepalive + stale detection | 2 |
| 5 | Close handshake (graceful + remote + timeout) | 3 |
| **Total** | | **12 tests** |

No new external dependencies. All logic in `session.rs`. Existing modules unchanged.

//! Sans-I/O session state machine for peer-to-peer connections.
//!
//! Each `Session` manages a single two-party encrypted connection:
//! identity-based handshake, resource ID mapping, keepalive, and
//! graceful close. The caller drives the session via events and
//! executes the returned actions.

use alloc::{string::String, vec, vec::Vec};
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

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
    /// Peer declared a resource (key expression <-> ExprId mapping).
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
    pub fn handle_event(&mut self, event: SessionEvent) -> Result<Vec<SessionAction>, ZenohError> {
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
        let signature: [u8; 64] = match proof.try_into() {
            Ok(sig) => sig,
            Err(_) => {
                self.state = SessionState::Closed;
                return Err(ZenohError::HandshakeFailed("invalid proof length".into()));
            }
        };

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
        self.state = SessionState::Active;
        Ok(vec![SessionAction::SessionOpened])
    }

    /// Declare a local resource, allocating the next ExprId.
    ///
    /// Only valid in the `Active` state.
    pub fn declare_resource(
        &mut self,
        key_expr: String,
    ) -> Result<(ExprId, Vec<SessionAction>), ZenohError> {
        if self.state != SessionState::Active {
            return Err(ZenohError::SessionNotActive);
        }
        let expr_id = self.next_expr_id;
        self.next_expr_id += 1;
        self.local_resources.insert(expr_id, key_expr.clone());
        let actions = vec![SessionAction::SendResourceDeclare { expr_id, key_expr }];
        Ok((expr_id, actions))
    }

    /// Undeclare a local resource.
    ///
    /// Only valid in the `Active` state.
    pub fn undeclare_resource(
        &mut self,
        expr_id: ExprId,
    ) -> Result<Vec<SessionAction>, ZenohError> {
        if self.state != SessionState::Active {
            return Err(ZenohError::SessionNotActive);
        }
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

    /// Initiate a graceful close. Transitions to Closing and emits `SendClose`.
    pub fn initiate_close(&mut self, now_ms: u64) -> Result<Vec<SessionAction>, ZenohError> {
        if self.state != SessionState::Active {
            return Err(ZenohError::SessionNotActive);
        }
        self.state = SessionState::Closing;
        self.close_initiated_ms = Some(now_ms);
        Ok(vec![SessionAction::SendClose])
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

    fn handle_timer_tick(&mut self, now_ms: u64) -> Result<Vec<SessionAction>, ZenohError> {
        let mut actions = Vec::new();
        self.last_tick_ms = now_ms;

        if self.state == SessionState::Closed {
            return Ok(actions);
        }

        // Check stale timeout (applies in Init, Active, and Closing)
        if now_ms.saturating_sub(self.last_received_ms) >= self.config.stale_timeout_ms {
            actions.push(SessionAction::PeerStale);
            actions.extend(self.force_close());
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

    fn handle_keepalive_received(&mut self) -> Result<Vec<SessionAction>, ZenohError> {
        self.last_received_ms = self.last_tick_ms;
        Ok(vec![])
    }

    fn handle_close_received(&mut self) -> Result<Vec<SessionAction>, ZenohError> {
        self.last_received_ms = self.last_tick_ms;
        let mut actions = vec![SessionAction::SendCloseAck];
        actions.extend(self.force_close());
        Ok(actions)
    }

    fn handle_close_ack_received(&mut self) -> Result<Vec<SessionAction>, ZenohError> {
        if self.state != SessionState::Closing {
            return Err(ZenohError::SessionNotActive);
        }
        self.last_received_ms = self.last_tick_ms;
        Ok(self.force_close())
    }
}

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

        let (alice_session, alice_actions) =
            Session::new(alice_id, bob_pub, SessionConfig::default(), 0);
        let (bob_session, bob_actions) =
            Session::new(bob_id, alice_pub, SessionConfig::default(), 0);

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
        assert!(matches!(
            &alice_actions[0],
            SessionAction::SendHandshake { .. }
        ));

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

        let (mut bob_session, _) = Session::new(bob_id, alice_pub, SessionConfig::default(), 0);

        // Eve signs a proof, but Bob expects Alice
        let eve_proof =
            eve_id.sign(&[b"harmony-session-v1" as &[u8], &bob_pub.address_hash].concat());

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
        let proof_for_bob =
            alice_id.sign(&[b"harmony-session-v1" as &[u8], &bob_pub.address_hash].concat());

        // Try to replay that proof to Charlie (who expects Alice)
        let (mut charlie_session, _) =
            Session::new(charlie_id, alice_pub, SessionConfig::default(), 0);

        // Charlie's address_hash != Bob's, so verification should fail
        let result = charlie_session.handle_event(SessionEvent::HandshakeReceived {
            proof: proof_for_bob.to_vec(),
        });
        assert!(result.is_err());
        assert_eq!(charlie_session.state(), SessionState::Closed);
    }

    #[test]
    fn handshake_invalid_proof_length_fails() {
        let (mut alice, _, _, _) = create_session_pair();
        let result = alice.handle_event(SessionEvent::HandshakeReceived {
            proof: vec![0u8; 32], // Wrong length
        });
        assert!(result.is_err());
        assert_eq!(alice.state(), SessionState::Closed);
    }

    #[test]
    fn resource_declare_undeclare_lifecycle() {
        let (mut alice, alice_actions, mut bob, bob_actions) = create_session_pair();
        complete_handshake(&mut alice, &alice_actions, &mut bob, &bob_actions);

        // Alice declares a local resource
        let (expr_id, actions) = alice
            .declare_resource("harmony/server/srv1/channel/general/msg".into())
            .unwrap();
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
        assert!(matches!(
            &actions[0],
            SessionAction::ResourceRemoved { expr_id: 1 }
        ));
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

        let (id1, _) = alice
            .declare_resource("harmony/server/a/msg".into())
            .unwrap();
        let (id2, _) = alice
            .declare_resource("harmony/server/b/msg".into())
            .unwrap();
        let (id3, _) = alice
            .declare_resource("harmony/server/c/msg".into())
            .unwrap();

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

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
        let actions = alice
            .handle_event(SessionEvent::TimerTick { now_ms: 50 })
            .unwrap();
        assert!(!actions.contains(&SessionAction::SendKeepalive));

        // At interval: keepalive emitted
        let actions = alice
            .handle_event(SessionEvent::TimerTick { now_ms: 100 })
            .unwrap();
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
        let actions = alice
            .handle_event(SessionEvent::TimerTick { now_ms: 200 })
            .unwrap();
        assert!(!actions.contains(&SessionAction::PeerStale));
        assert_eq!(alice.state(), SessionState::Active);

        // Stale
        let actions = alice
            .handle_event(SessionEvent::TimerTick { now_ms: 300 })
            .unwrap();
        assert!(actions.contains(&SessionAction::PeerStale));
        assert!(actions.contains(&SessionAction::SessionClosed));
        assert_eq!(alice.state(), SessionState::Closed);
    }

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
        let actions = alice.handle_event(SessionEvent::CloseAckReceived).unwrap();
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
        let actions = alice
            .handle_event(SessionEvent::TimerTick { now_ms: 30 })
            .unwrap();
        assert!(!actions.contains(&SessionAction::SessionClosed));

        // Timed out — force close
        let actions = alice
            .handle_event(SessionEvent::TimerTick { now_ms: 60 })
            .unwrap();
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
        let actions = bob.handle_event(SessionEvent::CloseReceived).unwrap();
        assert!(actions.contains(&SessionAction::SendCloseAck));
        assert!(actions.contains(&SessionAction::SessionClosed));
        assert_eq!(bob.state(), SessionState::Closed);
    }

    #[test]
    fn unsolicited_close_ack_in_active_rejected() {
        let (mut alice, alice_actions, mut bob, bob_actions) = create_session_pair();
        complete_handshake(&mut alice, &alice_actions, &mut bob, &bob_actions);

        // CloseAck without a preceding Close should be rejected
        let result = alice.handle_event(SessionEvent::CloseAckReceived);
        assert!(matches!(result, Err(ZenohError::SessionNotActive)));
        // Session stays Active — not torn down
        assert_eq!(alice.state(), SessionState::Active);
    }
}

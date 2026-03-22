use alloc::string::ToString;
use alloc::vec::Vec;
use harmony_crypto::aead::KEY_LENGTH;
use harmony_crypto::hash::blake3_hash;
use harmony_identity::{PqIdentity, PqPrivateIdentity};
use rand_core::CryptoRngCore;
use zeroize::Zeroize;

use crate::error::TunnelError;
use crate::event::{TunnelAction, TunnelEvent};
use crate::frame::{decrypt_frame, encrypt_frame, Frame, FrameTag};
use crate::handshake::{derive_session_keys, TunnelAccept, TunnelInit};

/// Base keepalive interval in milliseconds.
const KEEPALIVE_BASE_MS: u64 = 30_000;
/// Jitter range: ±5 seconds around the base.
const KEEPALIVE_JITTER_MS: u64 = 5_000;
/// Dead peer timeout: 3 missed keepalives at the maximum jittered interval.
/// With base=30s and jitter=5s, max interval is 35s; 3 × 35s = 105s.
/// Using the max-jitter value ensures a peer at worst-case jitter is never
/// declared dead before its third keepalive could have arrived.
const DEAD_TIMEOUT_MS: u64 = (KEEPALIVE_BASE_MS + KEEPALIVE_JITTER_MS) * 3; // 105_000ms

/// Tunnel session states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TunnelState {
    Initiating,
    Active,
    Closed,
}

/// Derive the 32-byte NodeId from an ML-DSA-65 public key.
pub fn node_id_from_dsa_pubkey(pubkey: &harmony_crypto::ml_dsa::MlDsaPublicKey) -> [u8; 32] {
    blake3_hash(&pubkey.as_bytes())
}

/// Sans-I/O state machine for a single tunnel connection.
pub struct TunnelSession {
    state: TunnelState,
    #[allow(dead_code)] // Used by future rekeying and diagnostics
    is_initiator: bool,
    /// Send key (our direction).
    send_key: [u8; KEY_LENGTH],
    /// Receive key (peer's direction).
    recv_key: [u8; KEY_LENGTH],
    /// Outbound nonce counter.
    send_nonce: u64,
    /// Inbound nonce counter.
    recv_nonce: u64,
    /// Remote peer's NodeId (BLAKE3 of their ML-DSA pubkey), used as AAD.
    remote_node_id: [u8; 32],
    /// Our NodeId, used as AAD for the remote's receive path.
    local_node_id: [u8; 32],
    /// Timestamp of last received data (for keepalive timeout).
    last_received_ms: u64,
    /// Timestamp of last sent data (for keepalive scheduling).
    last_sent_ms: u64,
    // -- Initiator-only state for completing handshake --
    /// Saved TunnelInit for transcript verification (initiator only).
    pending_init: Option<TunnelInit>,
    /// Saved shared secret for key derivation (initiator only, zeroized after use).
    pending_shared_secret: Option<Vec<u8>>,
    /// Initiator's nonce (saved for key derivation and potential rekeying).
    #[allow(dead_code)]
    init_nonce: [u8; 32],
    /// Expected responder's ML-DSA public key bytes (initiator only, for identity verification).
    expected_responder_pubkey: Option<Vec<u8>>,
}

impl Drop for TunnelSession {
    fn drop(&mut self) {
        self.send_key.zeroize();
        self.recv_key.zeroize();
        if let Some(ref mut ss) = self.pending_shared_secret {
            ss.zeroize();
        }
    }
}

impl TunnelSession {
    /// Returns the current state.
    pub fn state(&self) -> TunnelState {
        self.state
    }

    /// Create a new tunnel session as the **initiator**.
    ///
    /// Generates and returns the TunnelInit message as an `OutboundBytes` action.
    /// The session transitions to `Initiating` and waits for a TunnelAccept.
    pub fn new_initiator(
        rng: &mut impl CryptoRngCore,
        local_identity: &PqPrivateIdentity,
        remote_identity: &PqIdentity,
        now_ms: u64,
    ) -> Result<(Self, Vec<TunnelAction>), TunnelError> {
        let local_pub = local_identity.public_identity();
        let local_dsa_pk = &local_pub.verifying_key;
        let local_dsa_sk = local_identity.signing_key();
        let remote_kem_pk = &remote_identity.encryption_key;

        let (init_msg, shared_secret) =
            TunnelInit::create(rng, remote_kem_pk, local_dsa_pk, local_dsa_sk)?;

        let init_bytes = init_msg.to_bytes();
        let init_nonce = init_msg.nonce;

        let local_node_id = node_id_from_dsa_pubkey(local_dsa_pk);
        let remote_node_id = node_id_from_dsa_pubkey(&remote_identity.verifying_key);

        let session = Self {
            state: TunnelState::Initiating,
            is_initiator: true,
            send_key: [0u8; KEY_LENGTH],
            recv_key: [0u8; KEY_LENGTH],
            send_nonce: 0,
            recv_nonce: 0,
            remote_node_id,
            local_node_id,
            last_received_ms: now_ms,
            last_sent_ms: now_ms,
            pending_init: Some(init_msg),
            pending_shared_secret: Some(shared_secret.as_bytes().to_vec()),
            init_nonce,
            expected_responder_pubkey: Some(remote_identity.verifying_key.as_bytes()),
        };

        Ok((
            session,
            vec![TunnelAction::OutboundBytes { data: init_bytes }],
        ))
    }

    /// Create a new tunnel session as the **responder**.
    ///
    /// Processes the incoming TunnelInit, derives session keys, and returns
    /// the TunnelAccept as an `OutboundBytes` action plus a `HandshakeComplete`.
    pub fn new_responder(
        rng: &mut impl CryptoRngCore,
        local_identity: &PqPrivateIdentity,
        init_bytes: &[u8],
        now_ms: u64,
    ) -> Result<(Self, Vec<TunnelAction>), TunnelError> {
        let init_msg = TunnelInit::from_bytes(init_bytes)?;

        let local_pub = local_identity.public_identity();
        let local_dsa_pk = &local_pub.verifying_key;
        let local_dsa_sk = local_identity.signing_key();
        let local_kem_sk = local_identity.encryption_secret();

        let (accept_msg, shared_secret) =
            TunnelAccept::create(rng, local_kem_sk, local_dsa_pk, local_dsa_sk, &init_msg)?;

        let accept_bytes = accept_msg.to_bytes();

        // Derive directional keys
        let keys =
            derive_session_keys(shared_secret.as_bytes(), &init_msg.nonce, &accept_msg.nonce);

        let remote_node_id = node_id_from_dsa_pubkey(&init_msg.initiator_pubkey);
        let local_node_id = node_id_from_dsa_pubkey(local_dsa_pk);

        let session = Self {
            state: TunnelState::Active,
            is_initiator: false,
            // Responder sends on r2i, receives on i2r
            send_key: keys.r2i_key,
            recv_key: keys.i2r_key,
            send_nonce: 0,
            recv_nonce: 0,
            remote_node_id,
            local_node_id,
            last_received_ms: now_ms,
            last_sent_ms: now_ms,
            pending_init: None,
            pending_shared_secret: None,
            init_nonce: [0u8; 32],
            expected_responder_pubkey: None,
        };

        let actions = vec![
            TunnelAction::OutboundBytes { data: accept_bytes },
            TunnelAction::HandshakeComplete {
                peer_dsa_pubkey: init_msg.initiator_pubkey.as_bytes(),
                peer_node_id: remote_node_id,
            },
        ];

        Ok((session, actions))
    }

    /// Process a single event and return resulting actions.
    pub fn handle_event(&mut self, event: TunnelEvent) -> Result<Vec<TunnelAction>, TunnelError> {
        match event {
            TunnelEvent::InboundBytes { data, now_ms } => {
                self.last_received_ms = now_ms;
                self.handle_inbound(data)
            }
            TunnelEvent::SendReticulum { packet, now_ms } => {
                self.last_sent_ms = now_ms;
                self.handle_send(FrameTag::Reticulum, packet)
            }
            TunnelEvent::SendZenoh { message, now_ms } => {
                self.last_sent_ms = now_ms;
                self.handle_send(FrameTag::Zenoh, message)
            }
            TunnelEvent::Tick { now_ms } => self.handle_tick(now_ms),
            TunnelEvent::Close => self.handle_close(),
        }
    }

    fn handle_inbound(&mut self, data: Vec<u8>) -> Result<Vec<TunnelAction>, TunnelError> {
        match self.state {
            TunnelState::Initiating => self.handle_accept_response(data),
            TunnelState::Active => self.handle_encrypted_frame(data),
            _ => Err(TunnelError::InvalidState),
        }
    }

    /// Initiator processes TunnelAccept to complete handshake.
    fn handle_accept_response(&mut self, data: Vec<u8>) -> Result<Vec<TunnelAction>, TunnelError> {
        let accept_msg = TunnelAccept::from_bytes(&data)?;

        // Verify responder is the expected peer (MITM protection)
        if let Some(ref expected) = self.expected_responder_pubkey {
            if accept_msg.responder_pubkey.as_bytes() != *expected {
                return Err(TunnelError::SignatureVerificationFailed);
            }
        }

        let init_msg = self.pending_init.take().ok_or(TunnelError::InvalidState)?;

        // Verify transcript signature
        accept_msg.verify(&init_msg)?;

        // Derive session keys
        let mut shared_secret = self
            .pending_shared_secret
            .take()
            .ok_or(TunnelError::InvalidState)?;

        let keys = derive_session_keys(&shared_secret, &init_msg.nonce, &accept_msg.nonce);
        shared_secret.zeroize();

        // Initiator sends on i2r, receives on r2i
        self.send_key = keys.i2r_key;
        self.recv_key = keys.r2i_key;
        self.state = TunnelState::Active;

        let remote_node_id = node_id_from_dsa_pubkey(&accept_msg.responder_pubkey);
        self.remote_node_id = remote_node_id;

        Ok(vec![TunnelAction::HandshakeComplete {
            peer_dsa_pubkey: accept_msg.responder_pubkey.as_bytes(),
            peer_node_id: remote_node_id,
        }])
    }

    fn handle_encrypted_frame(&mut self, data: Vec<u8>) -> Result<Vec<TunnelAction>, TunnelError> {
        let frame = decrypt_frame(
            &data,
            &self.recv_key,
            &self.local_node_id,
            &mut self.recv_nonce,
        )?;

        match frame.tag {
            FrameTag::Keepalive => Ok(Vec::new()),
            FrameTag::Reticulum => Ok(vec![TunnelAction::ReticulumReceived {
                packet: frame.payload,
            }]),
            FrameTag::Zenoh => Ok(vec![TunnelAction::ZenohReceived {
                message: frame.payload,
            }]),
        }
    }

    fn handle_send(
        &mut self,
        tag: FrameTag,
        payload: Vec<u8>,
    ) -> Result<Vec<TunnelAction>, TunnelError> {
        if self.state != TunnelState::Active {
            return Err(TunnelError::InvalidState);
        }

        let frame = Frame { tag, payload };
        let encrypted = encrypt_frame(
            &frame,
            &self.send_key,
            &self.remote_node_id,
            &mut self.send_nonce,
        )?;

        Ok(vec![TunnelAction::OutboundBytes { data: encrypted }])
    }

    fn handle_tick(&mut self, now_ms: u64) -> Result<Vec<TunnelAction>, TunnelError> {
        if self.state != TunnelState::Active {
            return Ok(Vec::new());
        }

        let mut actions = Vec::new();

        // Check for dead peer
        if now_ms.saturating_sub(self.last_received_ms) >= DEAD_TIMEOUT_MS {
            self.state = TunnelState::Closed;
            actions.push(TunnelAction::Error {
                reason: "keepalive timeout".to_string(),
            });
            actions.push(TunnelAction::Closed);
            return Ok(actions);
        }

        // Jittered keepalive interval: 25-35 seconds.
        // Mix remote_node_id into the nonce seed so each session has a unique
        // cycle, preventing correlation of keepalive timing across sessions.
        let seed = self.send_nonce.wrapping_add(
            u64::from_le_bytes(self.remote_node_id[..8].try_into().unwrap())
        );
        let jitter = (seed % 11) * 1_000; // 0-10s, session-unique cycle
        let interval = KEEPALIVE_BASE_MS - KEEPALIVE_JITTER_MS + jitter;

        // Send keepalive if jittered interval elapsed
        if now_ms.saturating_sub(self.last_sent_ms) >= interval {
            // Nonce-derived padding: variable-length keepalives without needing RNG.
            // Zeros are fine — the payload is encrypted before transmission.
            let pad_seed = self.send_nonce.wrapping_add(
                u64::from_le_bytes(self.remote_node_id[..8].try_into().unwrap())
            );
            let pad_len = (pad_seed.wrapping_mul(7).wrapping_add(13) % 129) as usize;
            let padding = alloc::vec![0u8; pad_len];
            let frame = Frame {
                tag: FrameTag::Keepalive,
                payload: padding,
            };
            let encrypted = encrypt_frame(
                &frame,
                &self.send_key,
                &self.remote_node_id,
                &mut self.send_nonce,
            )?;
            self.last_sent_ms = now_ms;
            actions.push(TunnelAction::OutboundBytes { data: encrypted });
        }

        Ok(actions)
    }

    fn handle_close(&mut self) -> Result<Vec<TunnelAction>, TunnelError> {
        self.state = TunnelState::Closed;
        self.send_key.zeroize();
        self.recv_key.zeroize();
        Ok(vec![TunnelAction::Closed])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    /// Helper: create a paired initiator and responder identity.
    fn create_test_identities() -> (
        harmony_identity::PqPrivateIdentity,
        harmony_identity::PqPrivateIdentity,
    ) {
        let initiator = harmony_identity::PqPrivateIdentity::generate(&mut OsRng);
        let responder = harmony_identity::PqPrivateIdentity::generate(&mut OsRng);
        (initiator, responder)
    }

    #[test]
    fn initiator_emits_tunnel_init_on_creation() {
        let (initiator_id, responder_id) = create_test_identities();
        let responder_pub = responder_id.public_identity();

        let (session, actions) =
            TunnelSession::new_initiator(&mut OsRng, &initiator_id, responder_pub, 0).unwrap();

        assert_eq!(session.state(), TunnelState::Initiating);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], TunnelAction::OutboundBytes { .. }));
    }

    #[test]
    fn initiator_completes_handshake_on_valid_accept() {
        let (initiator_id, responder_id) = create_test_identities();
        let responder_pub = responder_id.public_identity();

        // Initiator sends TunnelInit
        let (mut initiator, init_actions) =
            TunnelSession::new_initiator(&mut OsRng, &initiator_id, responder_pub, 0).unwrap();

        // Extract the TunnelInit bytes
        let init_bytes = match &init_actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        // Responder processes TunnelInit and creates TunnelAccept
        let (responder, accept_actions) =
            TunnelSession::new_responder(&mut OsRng, &responder_id, &init_bytes, 0).unwrap();

        assert_eq!(responder.state(), TunnelState::Active);

        // Extract the TunnelAccept bytes
        let accept_bytes = accept_actions
            .iter()
            .find_map(|a| match a {
                TunnelAction::OutboundBytes { data } => Some(data.clone()),
                _ => None,
            })
            .expect("responder should emit OutboundBytes");

        // Initiator processes TunnelAccept
        let actions = initiator
            .handle_event(TunnelEvent::InboundBytes {
                data: accept_bytes,
                now_ms: 0,
            })
            .unwrap();

        assert_eq!(initiator.state(), TunnelState::Active);
        assert!(actions
            .iter()
            .any(|a| matches!(a, TunnelAction::HandshakeComplete { .. })));
    }

    #[test]
    fn paired_machines_exchange_data() {
        let (initiator_id, responder_id) = create_test_identities();
        let responder_pub = responder_id.public_identity().clone();

        // Complete handshake
        let (mut initiator, init_actions) =
            TunnelSession::new_initiator(&mut OsRng, &initiator_id, &responder_pub, 0).unwrap();

        let init_bytes = match &init_actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        let (mut responder, accept_actions) =
            TunnelSession::new_responder(&mut OsRng, &responder_id, &init_bytes, 0).unwrap();

        let accept_bytes = accept_actions
            .iter()
            .find_map(|a| match a {
                TunnelAction::OutboundBytes { data } => Some(data.clone()),
                _ => None,
            })
            .unwrap();

        initiator
            .handle_event(TunnelEvent::InboundBytes {
                data: accept_bytes,
                now_ms: 0,
            })
            .unwrap();

        // Initiator sends Reticulum packet to responder
        let actions = initiator
            .handle_event(TunnelEvent::SendReticulum {
                packet: b"hello-reticulum".to_vec(),
                now_ms: 0,
            })
            .unwrap();

        let encrypted = match &actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        let actions = responder
            .handle_event(TunnelEvent::InboundBytes {
                data: encrypted,
                now_ms: 0,
            })
            .unwrap();

        assert!(matches!(
            &actions[0],
            TunnelAction::ReticulumReceived { packet } if packet == b"hello-reticulum"
        ));

        // Responder sends Zenoh message back to initiator
        let actions = responder
            .handle_event(TunnelEvent::SendZenoh {
                message: b"hello-zenoh".to_vec(),
                now_ms: 0,
            })
            .unwrap();

        let encrypted = match &actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        let actions = initiator
            .handle_event(TunnelEvent::InboundBytes {
                data: encrypted,
                now_ms: 0,
            })
            .unwrap();

        assert!(matches!(
            &actions[0],
            TunnelAction::ZenohReceived { message } if message == b"hello-zenoh"
        ));
    }

    /// Helper: perform a full handshake and return (initiator, responder) both in Active state.
    fn complete_handshake() -> (
        TunnelSession,
        TunnelSession,
        harmony_identity::PqPrivateIdentity,
        harmony_identity::PqPrivateIdentity,
    ) {
        let (initiator_id, responder_id) = create_test_identities();
        let responder_pub = responder_id.public_identity().clone();

        let (mut initiator, init_actions) =
            TunnelSession::new_initiator(&mut OsRng, &initiator_id, &responder_pub, 0).unwrap();

        let init_bytes = match &init_actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        let (mut responder, accept_actions) =
            TunnelSession::new_responder(&mut OsRng, &responder_id, &init_bytes, 0).unwrap();

        let accept_bytes = accept_actions
            .iter()
            .find_map(|a| match a {
                TunnelAction::OutboundBytes { data } => Some(data.clone()),
                _ => None,
            })
            .unwrap();

        initiator
            .handle_event(TunnelEvent::InboundBytes {
                data: accept_bytes,
                now_ms: 0,
            })
            .unwrap();

        assert_eq!(initiator.state(), TunnelState::Active);
        assert_eq!(responder.state(), TunnelState::Active);

        (initiator, responder, initiator_id, responder_id)
    }

    // ── Task 6: Data Transfer and Adversarial Tests ──────────────────────────

    #[test]
    fn truncated_tunnel_init_rejected() {
        let (_initiator_id, responder_id) = create_test_identities();
        let result = TunnelSession::new_responder(&mut OsRng, &responder_id, &[0u8; 100], 0);
        assert!(result.is_err(), "expected error for truncated/garbage init");
    }

    #[test]
    fn wrong_responder_identity_rejected() {
        let (initiator_id, responder_id) = create_test_identities();
        let responder_pub = responder_id.public_identity().clone();

        // Initiator expects `responder_id` to respond
        let (mut initiator, init_actions) =
            TunnelSession::new_initiator(&mut OsRng, &initiator_id, &responder_pub, 0).unwrap();

        let init_bytes = match &init_actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        // Impersonator intercepts and replies with its own identity
        let impersonator_id = harmony_identity::PqPrivateIdentity::generate(&mut OsRng);
        let (_impersonator_session, accept_actions) =
            TunnelSession::new_responder(&mut OsRng, &impersonator_id, &init_bytes, 0).unwrap();

        let impersonator_accept_bytes = accept_actions
            .iter()
            .find_map(|a| match a {
                TunnelAction::OutboundBytes { data } => Some(data.clone()),
                _ => None,
            })
            .unwrap();

        // Initiator must reject — pubkey doesn't match expected responder
        let result = initiator.handle_event(TunnelEvent::InboundBytes {
            data: impersonator_accept_bytes,
            now_ms: 0,
        });
        assert!(
            result.is_err(),
            "initiator must reject impersonator's accept"
        );
    }

    #[test]
    fn send_before_handshake_fails() {
        let (initiator_id, responder_id) = create_test_identities();
        let responder_pub = responder_id.public_identity();

        let (mut initiator, _) =
            TunnelSession::new_initiator(&mut OsRng, &initiator_id, responder_pub, 0).unwrap();

        // State is Initiating — sending data must fail
        let result = initiator.handle_event(TunnelEvent::SendReticulum {
            packet: b"early".to_vec(),
            now_ms: 0,
        });
        assert!(
            result.is_err(),
            "SendReticulum before handshake must return error"
        );
        assert_eq!(initiator.state(), TunnelState::Initiating);
    }

    // ── Task 7: Keepalive and Timeout Tests ──────────────────────────────────

    #[test]
    fn keepalive_sent_after_jittered_interval() {
        let (mut initiator, _responder, _iid, _rid) = complete_handshake();

        // Session created with now_ms=0, so last_sent_ms=0.
        // Jitter is session-unique (mixed with remote_node_id), so the exact
        // interval is unpredictable. The range is always [25_000, 35_000]ms.

        // Tick at t=0: 0ms elapsed, no keepalive
        let actions = initiator
            .handle_event(TunnelEvent::Tick { now_ms: 0 })
            .unwrap();
        assert!(
            actions.is_empty(),
            "tick at t=0 with last_sent=0 should NOT send keepalive (0ms elapsed)"
        );

        // Tick at t=15000: only 15s elapsed → no keepalive (below minimum 25s)
        let actions = initiator
            .handle_event(TunnelEvent::Tick { now_ms: 15_000 })
            .unwrap();
        assert!(actions.is_empty(), "tick at 15s should not send keepalive");

        // Tick at t=35001: 35001ms >= max possible jittered interval (35000ms)
        // → keepalive must be sent regardless of session-specific jitter
        let actions = initiator
            .handle_event(TunnelEvent::Tick { now_ms: 35_001 })
            .unwrap();
        assert_eq!(
            actions.len(),
            1,
            "tick at 35001ms must emit exactly one keepalive (past max jitter)"
        );
        assert!(
            matches!(&actions[0], TunnelAction::OutboundBytes { .. }),
            "keepalive must be OutboundBytes"
        );
        assert_eq!(initiator.last_sent_ms, 35_001);
    }

    #[test]
    fn dead_peer_timeout() {
        let (mut initiator, _responder, _iid, _rid) = complete_handshake();

        // Session created with now_ms=0, so last_received_ms=0
        // Tick at t=105001: 105001 - 0 = 105001 >= DEAD_TIMEOUT_MS (105000)
        let actions = initiator
            .handle_event(TunnelEvent::Tick { now_ms: 105_001 })
            .unwrap();

        assert_eq!(initiator.state(), TunnelState::Closed);
        assert!(
            actions.iter().any(|a| matches!(a, TunnelAction::Closed)),
            "dead peer timeout must emit Closed"
        );
    }

    #[test]
    fn close_transitions_to_closed() {
        let (mut initiator, _responder, _iid, _rid) = complete_handshake();
        assert_eq!(initiator.state(), TunnelState::Active);

        let actions = initiator.handle_event(TunnelEvent::Close).unwrap();

        assert_eq!(initiator.state(), TunnelState::Closed);
        assert!(
            actions.iter().any(|a| matches!(a, TunnelAction::Closed)),
            "Close must emit TunnelAction::Closed"
        );
    }

    // ── Task 8: Timestamp Tracking ────────────────────────────────────────────

    #[test]
    fn inbound_data_resets_keepalive_timer() {
        let (mut initiator, mut responder, _iid, _rid) = complete_handshake();

        // Artificially set last_received_ms as if we last heard from peer at t=1000
        initiator.last_received_ms = 1000;

        // Responder sends a packet
        let send_actions = responder
            .handle_event(TunnelEvent::SendReticulum {
                packet: b"ping".to_vec(),
                now_ms: 0,
            })
            .unwrap();
        let encrypted = match &send_actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        // Initiator receives it at now_ms=50_000 — resets last_received_ms to 50_000
        initiator
            .handle_event(TunnelEvent::InboundBytes {
                data: encrypted,
                now_ms: 50_000,
            })
            .unwrap();
        assert_eq!(initiator.last_received_ms, 50_000);

        // Tick at 50000 + 104999 = 154999: 104999ms < DEAD_TIMEOUT_MS (105000) — NO timeout
        let actions = initiator
            .handle_event(TunnelEvent::Tick { now_ms: 154_999 })
            .unwrap();
        assert_ne!(
            initiator.state(),
            TunnelState::Closed,
            "should not timeout at 154999ms"
        );
        assert!(
            !actions.iter().any(|a| matches!(a, TunnelAction::Closed)),
            "must not be closed at 154999ms"
        );

        // Tick at 50000 + 105001 = 155001: 105001ms >= DEAD_TIMEOUT_MS (105000) — timeout
        let actions = initiator
            .handle_event(TunnelEvent::Tick { now_ms: 155_001 })
            .unwrap();
        assert_eq!(initiator.state(), TunnelState::Closed);
        assert!(
            actions.iter().any(|a| matches!(a, TunnelAction::Closed)),
            "must timeout at 155001ms"
        );
    }
}

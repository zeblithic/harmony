//! Reticulum link establishment — authenticated encrypted channels.
//!
//! Implements the three-step handshake:
//! 1. Initiator → Responder: Link request (ephemeral ECDH + Ed25519 keys)
//! 2. Responder → Initiator: Link proof (ECDH + identity signature)
//! 3. Initiator → Responder: RTT measurement (first encrypted message)
//!
//! After handshake, both sides share a 64-byte Fernet key derived via
//! `HKDF(ECDH_shared_secret, salt=link_id)`.

use ed25519_dalek::{SigningKey, VerifyingKey};
use harmony_crypto::{fernet, hash, hkdf};
use harmony_identity::identity::{Identity, PrivateIdentity, SIGNATURE_LENGTH};
use rand_core::CryptoRngCore;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
use zeroize::Zeroize;

use crate::context::PacketContext;
use crate::destination::DestinationName;
use crate::error::ReticulumError;
use crate::packet::{
    DestinationType, HeaderType, Packet, PacketFlags, PacketHeader, PacketType, PropagationType,
};

// ── Wire format constants ────────────────────────────────────────────

/// Link request data size: x25519_pub(32) + ed25519_pub(32) + signalling(3) = 67.
const LINK_REQUEST_SIZE: usize = 32 + 32 + 3;

/// Link proof data size: signature(64) + x25519_pub(32) + signalling(3) = 99.
const LINK_PROOF_SIZE: usize = SIGNATURE_LENGTH + 32 + 3;

/// Link identify data size: public_keys(64) + signature(64) = 128.
const LINK_IDENTIFY_SIZE: usize = 64 + SIGNATURE_LENGTH;

/// Link close data size: link_id(16).
const LINK_CLOSE_SIZE: usize = 16;

/// Signalling bytes size.
const SIGNALLING_SIZE: usize = 3;

// ── Signalling bytes ─────────────────────────────────────────────────

/// Cipher mode for link encryption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LinkMode {
    Aes256Cbc = 1,
}

/// Encode MTU and cipher mode into 3 signalling bytes.
///
/// `value = ((mode & 0x07) << 21) | (mtu & 0x1FFFFF)`
/// Returns the last 3 bytes of the big-endian u32.
pub fn encode_signalling(mtu: u32, mode: LinkMode) -> [u8; SIGNALLING_SIZE] {
    let value = ((mode as u32 & 0x07) << 21) | (mtu & 0x1F_FFFF);
    let be = value.to_be_bytes();
    [be[1], be[2], be[3]]
}

/// Decode 3 signalling bytes into (mtu, mode_bits).
#[cfg(test)]
fn decode_signalling(bytes: &[u8; SIGNALLING_SIZE]) -> (u32, u8) {
    let value = u32::from_be_bytes([0, bytes[0], bytes[1], bytes[2]]);
    let mtu = value & 0x1F_FFFF;
    let mode = ((value >> 21) & 0x07) as u8;
    (mtu, mode)
}

// ── Link state machine ──────────────────────────────────────────────

/// Link state in the handshake protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkState {
    /// Initiator has sent request, waiting for proof.
    Pending,
    /// Responder has sent proof, waiting for RTT.
    Handshake,
    /// Handshake complete, link is active.
    Active,
    /// Link has been closed.
    Closed,
}

/// An authenticated encrypted link between two Reticulum nodes.
pub struct Link {
    link_id: [u8; 16],
    state: LinkState,
    is_initiator: bool,
    /// Stored for future use (resource transfer, path requests).
    #[allow(dead_code)]
    destination_hash: [u8; 16],
    destination_identity: Identity,
    ephemeral_secret: Option<StaticSecret>,
    /// Stored for future use (initiator identification verification by responder).
    #[allow(dead_code)]
    peer_sig_pub: Option<VerifyingKey>,
    derived_key: Option<[u8; 64]>,
    remote_identity: Option<Identity>,
}

impl Drop for Link {
    fn drop(&mut self) {
        if let Some(ref mut key) = self.derived_key {
            key.zeroize();
        }
    }
}

impl Link {
    /// Get the link ID.
    pub fn link_id(&self) -> &[u8; 16] {
        &self.link_id
    }

    /// Get the current link state.
    pub fn state(&self) -> LinkState {
        self.state
    }

    /// Whether this side initiated the link.
    pub fn is_initiator(&self) -> bool {
        self.is_initiator
    }

    /// Get the remote identity, if identification has been performed.
    pub fn remote_identity(&self) -> Option<&Identity> {
        self.remote_identity.as_ref()
    }

    // ── Initiator flow ──────────────────────────────────────────────

    /// Create a link request to a destination.
    ///
    /// Returns the new link (in Pending state) and the request packet.
    pub fn initiate(
        rng: &mut impl CryptoRngCore,
        dest_identity: &Identity,
        dest_name: &DestinationName,
    ) -> Result<(Self, Packet), ReticulumError> {
        let destination_hash = dest_name.destination_hash(&dest_identity.address_hash);

        // Generate ephemeral X25519 keypair
        let eph_x25519_secret = StaticSecret::random_from_rng(&mut *rng);
        let eph_x25519_pub = X25519PublicKey::from(&eph_x25519_secret);

        // Generate ephemeral Ed25519 keypair
        let eph_ed25519_signing = SigningKey::generate(&mut *rng);
        let eph_ed25519_pub = eph_ed25519_signing.verifying_key();

        let signalling = encode_signalling(crate::packet::MTU as u32, LinkMode::Aes256Cbc);

        // Build request data: [x25519_pub:32][ed25519_pub:32][signalling:3]
        let mut data = Vec::with_capacity(LINK_REQUEST_SIZE);
        data.extend_from_slice(eph_x25519_pub.as_bytes());
        data.extend_from_slice(eph_ed25519_pub.as_bytes());
        data.extend_from_slice(&signalling);

        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::LinkRequest,
                },
                hops: 0,
                transport_id: None,
                destination_hash,
                context: PacketContext::None,
            },
            data,
        };

        // Compute link_id from hashable_part of the request
        let hp = packet.hashable_part()?;
        let link_id = hash::truncated_hash(&hp);

        let link = Self {
            link_id,
            state: LinkState::Pending,
            is_initiator: true,
            destination_hash,
            destination_identity: dest_identity.clone(),
            ephemeral_secret: Some(eph_x25519_secret),
            peer_sig_pub: None,
            derived_key: None,
            remote_identity: None,
        };

        // Note: eph_ed25519_signing is dropped here — initiator doesn't need it
        // after building the request. The verifying key is in the packet data.

        Ok((link, packet))
    }

    /// Complete the handshake after receiving a proof from the responder.
    ///
    /// Verifies the proof signature, performs ECDH, derives the session key,
    /// and builds the RTT packet.
    pub fn complete_handshake(
        &mut self,
        rng: &mut impl CryptoRngCore,
        proof_packet: &Packet,
        rtt_secs: f64,
    ) -> Result<Packet, ReticulumError> {
        if self.state == LinkState::Closed {
            return Err(ReticulumError::LinkAlreadyClosed);
        }

        let data = &proof_packet.data;
        if data.len() < LINK_PROOF_SIZE {
            return Err(ReticulumError::LinkProofTooShort {
                minimum: LINK_PROOF_SIZE,
                actual: data.len(),
            });
        }

        // Parse proof: [signature:64][responder_x25519_pub:32][signalling:3]
        let signature: [u8; SIGNATURE_LENGTH] = data[..64].try_into().unwrap();
        let responder_x25519_bytes: [u8; 32] = data[64..96].try_into().unwrap();
        let _signalling: [u8; SIGNALLING_SIZE] = data[96..99].try_into().unwrap();

        let responder_x25519_pub = X25519PublicKey::from(responder_x25519_bytes);

        // Build the signed data that the responder signed:
        // link_id || responder_x25519_pub || destination_ed25519_pub || signalling
        let mut signed_data = Vec::with_capacity(16 + 32 + 32 + SIGNALLING_SIZE);
        signed_data.extend_from_slice(&self.link_id);
        signed_data.extend_from_slice(&responder_x25519_bytes);
        signed_data.extend_from_slice(self.destination_identity.verifying_key.as_bytes());
        signed_data.extend_from_slice(&_signalling);

        // Verify signature using destination's Ed25519 key
        self.destination_identity
            .verify(&signed_data, &signature)
            .map_err(|_| ReticulumError::LinkProofSignatureInvalid)?;

        // ECDH: our ephemeral secret × responder's X25519 pub
        let eph_secret = self
            .ephemeral_secret
            .take()
            .ok_or(ReticulumError::LinkNotActive)?;
        let shared_secret = eph_secret.diffie_hellman(&responder_x25519_pub);

        // Derive 64-byte Fernet key
        let derived_key = hkdf::derive_key_256(shared_secret.as_bytes(), Some(&self.link_id));
        self.derived_key = Some(derived_key);
        self.state = LinkState::Active;

        // Build RTT packet: encrypt msgpack float64
        let rtt_data = build_rtt_data(rtt_secs);
        let encrypted = fernet::encrypt(rng, &derived_key, &rtt_data)
            .map_err(|e| ReticulumError::Identity(e.into()))?;

        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: true,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Link,
                    packet_type: PacketType::Data,
                },
                hops: 0,
                transport_id: None,
                destination_hash: self.link_id,
                context: PacketContext::Lrrtt,
            },
            data: encrypted,
        };

        Ok(packet)
    }

    // ── Responder flow ──────────────────────────────────────────────

    /// Receive a link request and build a proof response.
    ///
    /// Returns the new link (in Handshake state) and the proof packet.
    pub fn respond(
        _rng: &mut impl CryptoRngCore,
        private_identity: &PrivateIdentity,
        dest_name: &DestinationName,
        request_packet: &Packet,
    ) -> Result<(Self, Packet), ReticulumError> {
        let data = &request_packet.data;
        if data.len() < LINK_REQUEST_SIZE {
            return Err(ReticulumError::LinkRequestTooShort {
                minimum: LINK_REQUEST_SIZE,
                actual: data.len(),
            });
        }

        // Parse request: [initiator_x25519_pub:32][initiator_ed25519_pub:32][signalling:3]
        let initiator_x25519_bytes: [u8; 32] = data[..32].try_into().unwrap();
        let initiator_ed25519_bytes: [u8; 32] = data[32..64].try_into().unwrap();
        let _req_signalling: [u8; SIGNALLING_SIZE] = data[64..67].try_into().unwrap();

        let initiator_x25519_pub = X25519PublicKey::from(initiator_x25519_bytes);
        let initiator_ed25519_pub = VerifyingKey::from_bytes(&initiator_ed25519_bytes)
            .map_err(|_| ReticulumError::LinkProofSignatureInvalid)?;

        // Compute link_id from hashable_part of the request
        let hp = request_packet.hashable_part()?;
        let link_id = hash::truncated_hash(&hp);

        let identity = private_identity.public_identity();
        let destination_hash = dest_name.destination_hash(&identity.address_hash);

        // ECDH: our identity secret × initiator's ephemeral X25519 pub
        let shared_secret = private_identity.ecdh(&initiator_x25519_pub);

        // Derive 64-byte Fernet key
        let derived_key = hkdf::derive_key_256(shared_secret.as_bytes(), Some(&link_id));

        // Build proof signalling
        let signalling = encode_signalling(crate::packet::MTU as u32, LinkMode::Aes256Cbc);

        // Sign: link_id || our_x25519_pub || our_ed25519_pub || signalling
        let mut signed_data = Vec::with_capacity(16 + 32 + 32 + SIGNALLING_SIZE);
        signed_data.extend_from_slice(&link_id);
        signed_data.extend_from_slice(identity.encryption_key.as_bytes());
        signed_data.extend_from_slice(identity.verifying_key.as_bytes());
        signed_data.extend_from_slice(&signalling);

        let signature = private_identity.sign(&signed_data);

        // Build proof data: [signature:64][our_x25519_pub:32][signalling:3]
        let mut proof_data = Vec::with_capacity(LINK_PROOF_SIZE);
        proof_data.extend_from_slice(&signature);
        proof_data.extend_from_slice(identity.encryption_key.as_bytes());
        proof_data.extend_from_slice(&signalling);

        let proof_packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: true,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Link,
                    packet_type: PacketType::Proof,
                },
                hops: 0,
                transport_id: None,
                destination_hash: link_id,
                context: PacketContext::LrProof,
            },
            data: proof_data,
        };

        let link = Self {
            link_id,
            state: LinkState::Handshake,
            is_initiator: false,
            destination_hash,
            destination_identity: identity.clone(),
            ephemeral_secret: None,
            peer_sig_pub: Some(initiator_ed25519_pub),
            derived_key: Some(derived_key),
            remote_identity: None,
        };

        Ok((link, proof_packet))
    }

    /// Activate the link after receiving the RTT packet from the initiator.
    ///
    /// Returns the RTT value in seconds.
    pub fn activate(&mut self, rtt_packet: &Packet) -> Result<f64, ReticulumError> {
        if self.state == LinkState::Closed {
            return Err(ReticulumError::LinkAlreadyClosed);
        }

        let key = self.derived_key.as_ref().ok_or(ReticulumError::LinkNotActive)?;
        let plaintext = fernet::decrypt(key, &rtt_packet.data)
            .map_err(|e| ReticulumError::Identity(e.into()))?;

        let rtt = parse_rtt_data(&plaintext)?;
        self.state = LinkState::Active;
        Ok(rtt)
    }

    // ── Data operations ─────────────────────────────────────────────

    /// Encrypt plaintext using the link's session key.
    pub fn encrypt(
        &self,
        rng: &mut impl CryptoRngCore,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, ReticulumError> {
        self.require_active()?;
        let key = self.derived_key.as_ref().unwrap();
        fernet::encrypt(rng, key, plaintext).map_err(|e| ReticulumError::Identity(e.into()))
    }

    /// Decrypt ciphertext using the link's session key.
    pub fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, ReticulumError> {
        self.require_active()?;
        let key = self.derived_key.as_ref().unwrap();
        fernet::decrypt(key, ciphertext).map_err(|e| ReticulumError::Identity(e.into()))
    }

    // ── Identification ──────────────────────────────────────────────

    /// Build a link identification packet (initiator → responder).
    ///
    /// Sends the initiator's real identity, signed over link_id.
    pub fn build_identify(
        &self,
        rng: &mut impl CryptoRngCore,
        private_identity: &PrivateIdentity,
    ) -> Result<Packet, ReticulumError> {
        self.require_active()?;
        let key = self.derived_key.as_ref().unwrap();

        let pub_keys = private_identity.public_identity().to_public_bytes();

        // Sign: link_id || identity_public_keys
        let mut signed_data = Vec::with_capacity(16 + 64);
        signed_data.extend_from_slice(&self.link_id);
        signed_data.extend_from_slice(&pub_keys);
        let signature = private_identity.sign(&signed_data);

        // Identify data: [public_keys:64][signature:64]
        let mut plaintext = Vec::with_capacity(LINK_IDENTIFY_SIZE);
        plaintext.extend_from_slice(&pub_keys);
        plaintext.extend_from_slice(&signature);

        let encrypted = fernet::encrypt(rng, key, &plaintext)
            .map_err(|e| ReticulumError::Identity(e.into()))?;

        Ok(Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: true,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Link,
                    packet_type: PacketType::Data,
                },
                hops: 0,
                transport_id: None,
                destination_hash: self.link_id,
                context: PacketContext::LinkIdentify,
            },
            data: encrypted,
        })
    }

    /// Validate an identification packet and store the remote identity.
    pub fn validate_identification(
        &mut self,
        packet: &Packet,
    ) -> Result<Identity, ReticulumError> {
        self.require_active()?;
        let key = self.derived_key.as_ref().unwrap();

        let plaintext = fernet::decrypt(key, &packet.data)
            .map_err(|_| ReticulumError::LinkIdentificationInvalid)?;

        if plaintext.len() < LINK_IDENTIFY_SIZE {
            return Err(ReticulumError::LinkIdentificationInvalid);
        }

        let pub_keys: [u8; 64] = plaintext[..64].try_into().unwrap();
        let signature: [u8; SIGNATURE_LENGTH] = plaintext[64..128].try_into().unwrap();

        let identity = Identity::from_public_bytes(&pub_keys)
            .map_err(|_| ReticulumError::LinkIdentificationInvalid)?;

        // Verify: signature covers link_id || public_keys
        let mut signed_data = Vec::with_capacity(16 + 64);
        signed_data.extend_from_slice(&self.link_id);
        signed_data.extend_from_slice(&pub_keys);

        identity
            .verify(&signed_data, &signature)
            .map_err(|_| ReticulumError::LinkIdentificationInvalid)?;

        self.remote_identity = Some(identity.clone());
        Ok(identity)
    }

    // ── Close ───────────────────────────────────────────────────────

    /// Build a link close packet.
    pub fn build_close(
        &self,
        rng: &mut impl CryptoRngCore,
    ) -> Result<Packet, ReticulumError> {
        self.require_active()?;
        let key = self.derived_key.as_ref().unwrap();

        let encrypted = fernet::encrypt(rng, key, &self.link_id)
            .map_err(|e| ReticulumError::Identity(e.into()))?;

        Ok(Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: true,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Link,
                    packet_type: PacketType::Data,
                },
                hops: 0,
                transport_id: None,
                destination_hash: self.link_id,
                context: PacketContext::LinkClose,
            },
            data: encrypted,
        })
    }

    /// Process a received close packet.
    pub fn receive_close(&mut self, packet: &Packet) -> Result<(), ReticulumError> {
        self.require_active()?;
        let key = self.derived_key.as_ref().unwrap();

        let plaintext = fernet::decrypt(key, &packet.data)
            .map_err(|e| ReticulumError::Identity(e.into()))?;

        if plaintext.len() < LINK_CLOSE_SIZE {
            return Err(ReticulumError::LinkNotActive);
        }

        let received_link_id: [u8; 16] = plaintext[..16].try_into().unwrap();
        if received_link_id != self.link_id {
            return Err(ReticulumError::LinkNotActive);
        }

        self.state = LinkState::Closed;
        Ok(())
    }

    // ── Keepalive ───────────────────────────────────────────────────

    /// Build a keepalive packet.
    pub fn build_keepalive(&self) -> Result<Packet, ReticulumError> {
        self.require_active()?;

        Ok(Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: true,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Link,
                    packet_type: PacketType::Data,
                },
                hops: 0,
                transport_id: None,
                destination_hash: self.link_id,
                context: PacketContext::Keepalive,
            },
            data: vec![0xFF],
        })
    }

    // ── Helpers ──────────────────────────────────────────────────────

    fn require_active(&self) -> Result<(), ReticulumError> {
        match self.state {
            LinkState::Active => Ok(()),
            LinkState::Closed => Err(ReticulumError::LinkAlreadyClosed),
            _ => Err(ReticulumError::LinkNotActive),
        }
    }
}

// ── RTT data encoding ────────────────────────────────────────────────

/// Encode RTT as msgpack float64: [0xcb][f64.to_be_bytes()].
fn build_rtt_data(rtt_secs: f64) -> [u8; 9] {
    let mut buf = [0u8; 9];
    buf[0] = 0xcb;
    buf[1..].copy_from_slice(&rtt_secs.to_be_bytes());
    buf
}

/// Parse RTT from msgpack float64.
fn parse_rtt_data(data: &[u8]) -> Result<f64, ReticulumError> {
    if data.len() < 9 || data[0] != 0xcb {
        return Err(ReticulumError::LinkNotActive);
    }
    let bytes: [u8; 8] = data[1..9].try_into().unwrap();
    Ok(f64::from_be_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    // ── Signalling encode/decode ────────────────────────────────────

    #[test]
    fn signalling_roundtrip() {
        let bytes = encode_signalling(500, LinkMode::Aes256Cbc);
        let (mtu, mode) = decode_signalling(&bytes);
        assert_eq!(mtu, 500);
        assert_eq!(mode, LinkMode::Aes256Cbc as u8);
    }

    #[test]
    fn signalling_mtu_500_aes256cbc() {
        let bytes = encode_signalling(500, LinkMode::Aes256Cbc);
        let (mtu, mode) = decode_signalling(&bytes);
        assert_eq!(mtu, 500);
        assert_eq!(mode, 1);
    }

    #[test]
    fn signalling_max_mtu() {
        // Maximum 21-bit value = 0x1FFFFF = 2097151
        let max_mtu: u32 = 0x1F_FFFF;
        let bytes = encode_signalling(max_mtu, LinkMode::Aes256Cbc);
        let (mtu, mode) = decode_signalling(&bytes);
        assert_eq!(mtu, max_mtu);
        assert_eq!(mode, 1);
    }

    // ── RTT data encoding ───────────────────────────────────────────

    #[test]
    fn rtt_data_roundtrip() {
        let rtt = 0.042;
        let data = build_rtt_data(rtt);
        assert_eq!(data[0], 0xcb);
        let parsed = parse_rtt_data(&data).unwrap();
        assert!((parsed - rtt).abs() < f64::EPSILON);
    }

    #[test]
    fn rtt_data_invalid_prefix() {
        let data = [0x00; 9];
        assert!(parse_rtt_data(&data).is_err());
    }

    #[test]
    fn rtt_data_too_short() {
        let data = [0xcb, 0x00];
        assert!(parse_rtt_data(&data).is_err());
    }

    // ── Full handshake roundtrip ────────────────────────────────────

    fn setup_handshake() -> (PrivateIdentity, DestinationName) {
        let responder_identity = PrivateIdentity::generate(&mut OsRng);
        let dest_name = DestinationName::from_name("testapp", &["link"]).unwrap();
        (responder_identity, dest_name)
    }

    #[test]
    fn handshake_roundtrip() {
        let (responder_priv, dest_name) = setup_handshake();
        let responder_pub = responder_priv.public_identity();

        // Step 1: Initiator creates link request
        let (mut initiator_link, request_packet) =
            Link::initiate(&mut OsRng, responder_pub, &dest_name).unwrap();
        assert_eq!(initiator_link.state(), LinkState::Pending);
        assert!(initiator_link.is_initiator());

        // Step 2: Responder receives request and builds proof
        let (mut responder_link, proof_packet) =
            Link::respond(&mut OsRng, &responder_priv, &dest_name, &request_packet).unwrap();
        assert_eq!(responder_link.state(), LinkState::Handshake);
        assert!(!responder_link.is_initiator());

        // Both sides should have the same link_id
        assert_eq!(initiator_link.link_id(), responder_link.link_id());

        // Step 3: Initiator verifies proof and sends RTT
        let rtt_packet = initiator_link
            .complete_handshake(&mut OsRng, &proof_packet, 0.042)
            .unwrap();
        assert_eq!(initiator_link.state(), LinkState::Active);

        // Step 4: Responder receives RTT and activates
        let rtt = responder_link.activate(&rtt_packet).unwrap();
        assert_eq!(responder_link.state(), LinkState::Active);
        assert!((rtt - 0.042).abs() < f64::EPSILON);
    }

    #[test]
    fn both_sides_derive_same_key() {
        let (responder_priv, dest_name) = setup_handshake();
        let responder_pub = responder_priv.public_identity();

        let (mut initiator_link, request_packet) =
            Link::initiate(&mut OsRng, responder_pub, &dest_name).unwrap();
        let (responder_link, proof_packet) =
            Link::respond(&mut OsRng, &responder_priv, &dest_name, &request_packet).unwrap();

        initiator_link
            .complete_handshake(&mut OsRng, &proof_packet, 0.01)
            .unwrap();

        // Both should have the same derived key
        assert_eq!(
            initiator_link.derived_key.as_ref().unwrap(),
            responder_link.derived_key.as_ref().unwrap()
        );
    }

    // ── Encrypt/decrypt roundtrip ───────────────────────────────────

    fn setup_active_link() -> (Link, Link) {
        let (responder_priv, dest_name) = setup_handshake();
        let responder_pub = responder_priv.public_identity();

        let (mut initiator_link, request_packet) =
            Link::initiate(&mut OsRng, responder_pub, &dest_name).unwrap();
        let (mut responder_link, proof_packet) =
            Link::respond(&mut OsRng, &responder_priv, &dest_name, &request_packet).unwrap();

        let rtt_packet = initiator_link
            .complete_handshake(&mut OsRng, &proof_packet, 0.01)
            .unwrap();
        responder_link.activate(&rtt_packet).unwrap();

        (initiator_link, responder_link)
    }

    #[test]
    fn encrypt_decrypt_roundtrip_initiator_to_responder() {
        let (initiator, responder) = setup_active_link();
        let plaintext = b"hello from initiator";

        let encrypted = initiator.encrypt(&mut OsRng, plaintext).unwrap();
        let decrypted = responder.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypt_decrypt_roundtrip_responder_to_initiator() {
        let (initiator, responder) = setup_active_link();
        let plaintext = b"hello from responder";

        let encrypted = responder.encrypt(&mut OsRng, plaintext).unwrap();
        let decrypted = initiator.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypt_decrypt_empty_plaintext() {
        let (initiator, responder) = setup_active_link();

        let encrypted = initiator.encrypt(&mut OsRng, b"").unwrap();
        let decrypted = responder.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, b"");
    }

    #[test]
    fn encrypt_decrypt_large_payload() {
        let (initiator, responder) = setup_active_link();
        let plaintext = vec![0xAB; 4096];

        let encrypted = initiator.encrypt(&mut OsRng, &plaintext).unwrap();
        let decrypted = responder.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    // ── Identification ──────────────────────────────────────────────

    #[test]
    fn identification_roundtrip() {
        let initiator_priv = PrivateIdentity::generate(&mut OsRng);
        let (initiator, mut responder) = setup_active_link();

        let identify_packet = initiator
            .build_identify(&mut OsRng, &initiator_priv)
            .unwrap();

        let remote = responder.validate_identification(&identify_packet).unwrap();
        assert_eq!(&remote, initiator_priv.public_identity());
        assert_eq!(
            responder.remote_identity().unwrap(),
            initiator_priv.public_identity()
        );
    }

    #[test]
    fn identification_tampered_signature_rejected() {
        let initiator_priv = PrivateIdentity::generate(&mut OsRng);
        let (initiator, mut responder) = setup_active_link();

        let mut identify_packet = initiator
            .build_identify(&mut OsRng, &initiator_priv)
            .unwrap();

        // Tamper with encrypted data
        if let Some(byte) = identify_packet.data.last_mut() {
            *byte ^= 0x01;
        }

        assert!(responder.validate_identification(&identify_packet).is_err());
    }

    #[test]
    fn identification_wrong_identity_rejected() {
        let wrong_identity = PrivateIdentity::generate(&mut OsRng);
        let (initiator, mut responder) = setup_active_link();

        // Manually build identify with wrong signing key but valid encryption
        // The identify will decrypt OK but signature won't match
        let key = initiator.derived_key.as_ref().unwrap();
        let pub_keys = wrong_identity.public_identity().to_public_bytes();

        // Sign with correct key material but different identity keys
        let another_priv = PrivateIdentity::generate(&mut OsRng);
        let mut signed_data = Vec::with_capacity(16 + 64);
        signed_data.extend_from_slice(&initiator.link_id);
        signed_data.extend_from_slice(&pub_keys);
        let signature = another_priv.sign(&signed_data); // wrong signer!

        let mut plaintext = Vec::with_capacity(LINK_IDENTIFY_SIZE);
        plaintext.extend_from_slice(&pub_keys);
        plaintext.extend_from_slice(&signature);

        let encrypted = fernet::encrypt(&mut OsRng, key, &plaintext).unwrap();

        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: true,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Link,
                    packet_type: PacketType::Data,
                },
                hops: 0,
                transport_id: None,
                destination_hash: initiator.link_id,
                context: PacketContext::LinkIdentify,
            },
            data: encrypted,
        };

        assert!(matches!(
            responder.validate_identification(&packet),
            Err(ReticulumError::LinkIdentificationInvalid)
        ));
    }

    // ── Close ───────────────────────────────────────────────────────

    #[test]
    fn close_roundtrip() {
        let (initiator, mut responder) = setup_active_link();

        let close_packet = initiator.build_close(&mut OsRng).unwrap();
        responder.receive_close(&close_packet).unwrap();
        assert_eq!(responder.state(), LinkState::Closed);
    }

    #[test]
    fn close_wrong_link_id_rejected() {
        let (initiator, mut responder) = setup_active_link();

        // Manually build a close with wrong link_id
        let key = initiator.derived_key.as_ref().unwrap();
        let wrong_id = [0xFF; 16];
        let encrypted = fernet::encrypt(&mut OsRng, key, &wrong_id).unwrap();

        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: true,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Link,
                    packet_type: PacketType::Data,
                },
                hops: 0,
                transport_id: None,
                destination_hash: initiator.link_id,
                context: PacketContext::LinkClose,
            },
            data: encrypted,
        };

        assert!(responder.receive_close(&packet).is_err());
    }

    // ── Keepalive ───────────────────────────────────────────────────

    #[test]
    fn keepalive_packet() {
        let (initiator, _) = setup_active_link();

        let keepalive = initiator.build_keepalive().unwrap();
        assert_eq!(keepalive.header.context, PacketContext::Keepalive);
        assert_eq!(keepalive.data, vec![0xFF]);
        assert_eq!(keepalive.header.destination_hash, *initiator.link_id());
    }

    // ── State guards ────────────────────────────────────────────────

    #[test]
    fn encrypt_before_active_rejected() {
        let (responder_priv, dest_name) = setup_handshake();
        let responder_pub = responder_priv.public_identity();

        let (initiator, _request) =
            Link::initiate(&mut OsRng, responder_pub, &dest_name).unwrap();

        // Link is Pending, not Active
        assert!(matches!(
            initiator.encrypt(&mut OsRng, b"hello"),
            Err(ReticulumError::LinkNotActive)
        ));
    }

    #[test]
    fn operations_after_close_rejected() {
        let (initiator, mut responder) = setup_active_link();

        let close_packet = initiator.build_close(&mut OsRng).unwrap();
        responder.receive_close(&close_packet).unwrap();

        // All operations should fail on closed link
        assert!(matches!(
            responder.encrypt(&mut OsRng, b"hello"),
            Err(ReticulumError::LinkAlreadyClosed)
        ));
        assert!(matches!(
            responder.decrypt(&[0; 64]),
            Err(ReticulumError::LinkAlreadyClosed)
        ));
        assert!(matches!(
            responder.build_close(&mut OsRng),
            Err(ReticulumError::LinkAlreadyClosed)
        ));
        assert!(matches!(
            responder.build_keepalive(),
            Err(ReticulumError::LinkAlreadyClosed)
        ));
    }

    // ── Link request too short ──────────────────────────────────────

    #[test]
    fn link_request_too_short_rejected() {
        let responder_priv = PrivateIdentity::generate(&mut OsRng);
        let dest_name = DestinationName::from_name("app", &[]).unwrap();

        let short_packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::LinkRequest,
                },
                hops: 0,
                transport_id: None,
                destination_hash: [0; 16],
                context: PacketContext::None,
            },
            data: vec![0; 30], // Too short
        };

        assert!(matches!(
            Link::respond(&mut OsRng, &responder_priv, &dest_name, &short_packet),
            Err(ReticulumError::LinkRequestTooShort { minimum: 67, actual: 30 })
        ));
    }

    // ── Link proof too short ────────────────────────────────────────

    #[test]
    fn link_proof_too_short_rejected() {
        let (responder_priv, dest_name) = setup_handshake();
        let responder_pub = responder_priv.public_identity();

        let (mut initiator, _request) =
            Link::initiate(&mut OsRng, responder_pub, &dest_name).unwrap();

        let short_proof = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: true,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Link,
                    packet_type: PacketType::Proof,
                },
                hops: 0,
                transport_id: None,
                destination_hash: *initiator.link_id(),
                context: PacketContext::LrProof,
            },
            data: vec![0; 50], // Too short
        };

        assert!(matches!(
            initiator.complete_handshake(&mut OsRng, &short_proof, 0.01),
            Err(ReticulumError::LinkProofTooShort { minimum: 99, actual: 50 })
        ));
    }

    // ── Proof signature verification ────────────────────────────────

    #[test]
    fn tampered_proof_signature_rejected() {
        let (responder_priv, dest_name) = setup_handshake();
        let responder_pub = responder_priv.public_identity();

        let (mut initiator, request) =
            Link::initiate(&mut OsRng, responder_pub, &dest_name).unwrap();
        let (_, mut proof_packet) =
            Link::respond(&mut OsRng, &responder_priv, &dest_name, &request).unwrap();

        // Tamper with signature in proof data
        proof_packet.data[0] ^= 0x01;

        assert!(matches!(
            initiator.complete_handshake(&mut OsRng, &proof_packet, 0.01),
            Err(ReticulumError::LinkProofSignatureInvalid)
        ));
    }
}

use alloc::vec::Vec;
use harmony_crypto::ml_dsa::{MlDsaPublicKey, MlDsaSignature};
use harmony_crypto::ml_kem::MlKemPublicKey;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyEvent {
    Inception(InceptionEvent),
    Rotation(RotationEvent),
    Interaction(InteractionEvent),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InceptionEvent {
    pub signing_key: MlDsaPublicKey,
    pub encryption_key: MlKemPublicKey,
    pub next_key_commitment: [u8; 32],
    pub created_at: u64,
    pub signature: MlDsaSignature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationEvent {
    pub sequence: u64,
    pub previous_hash: [u8; 32],
    pub signing_key: MlDsaPublicKey,
    pub encryption_key: MlKemPublicKey,
    pub next_key_commitment: [u8; 32],
    pub created_at: u64,
    pub old_signature: MlDsaSignature,
    pub new_signature: MlDsaSignature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    pub sequence: u64,
    pub previous_hash: [u8; 32],
    pub data_hash: [u8; 32],
    pub created_at: u64,
    pub signature: MlDsaSignature,
}

impl KeyEvent {
    pub fn sequence(&self) -> u64 {
        match self {
            KeyEvent::Inception(_) => 0,
            KeyEvent::Rotation(e) => e.sequence,
            KeyEvent::Interaction(e) => e.sequence,
        }
    }
}

/// Serialize the unsigned inception payload from individual field values.
///
/// This is the canonical implementation. Use this when constructing an inception
/// event, before the struct exists, so the payload can be signed.
pub fn serialize_inception_payload_parts(
    signing_key: &MlDsaPublicKey,
    encryption_key: &MlKemPublicKey,
    next_key_commitment: &[u8; 32],
    created_at: u64,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&signing_key.as_bytes());
    buf.extend_from_slice(&encryption_key.as_bytes());
    buf.extend_from_slice(next_key_commitment);
    buf.extend_from_slice(&created_at.to_le_bytes());
    buf
}

pub fn serialize_inception_payload(event: &InceptionEvent) -> Vec<u8> {
    serialize_inception_payload_parts(
        &event.signing_key,
        &event.encryption_key,
        &event.next_key_commitment,
        event.created_at,
    )
}

/// Serialize the unsigned rotation payload from individual field values.
pub fn serialize_rotation_payload_parts(
    sequence: u64,
    previous_hash: &[u8; 32],
    signing_key: &MlDsaPublicKey,
    encryption_key: &MlKemPublicKey,
    next_key_commitment: &[u8; 32],
    created_at: u64,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&sequence.to_le_bytes());
    buf.extend_from_slice(previous_hash);
    buf.extend_from_slice(&signing_key.as_bytes());
    buf.extend_from_slice(&encryption_key.as_bytes());
    buf.extend_from_slice(next_key_commitment);
    buf.extend_from_slice(&created_at.to_le_bytes());
    buf
}

pub fn serialize_rotation_payload(event: &RotationEvent) -> Vec<u8> {
    serialize_rotation_payload_parts(
        event.sequence,
        &event.previous_hash,
        &event.signing_key,
        &event.encryption_key,
        &event.next_key_commitment,
        event.created_at,
    )
}

/// Serialize the unsigned interaction payload from individual field values.
pub fn serialize_interaction_payload_parts(
    sequence: u64,
    previous_hash: &[u8; 32],
    data_hash: &[u8; 32],
    created_at: u64,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&sequence.to_le_bytes());
    buf.extend_from_slice(previous_hash);
    buf.extend_from_slice(data_hash);
    buf.extend_from_slice(&created_at.to_le_bytes());
    buf
}

pub fn serialize_interaction_payload(event: &InteractionEvent) -> Vec<u8> {
    serialize_interaction_payload_parts(
        event.sequence,
        &event.previous_hash,
        &event.data_hash,
        event.created_at,
    )
}

pub fn serialize_event_payload(event: &KeyEvent) -> Vec<u8> {
    match event {
        KeyEvent::Inception(e) => serialize_inception_payload(e),
        KeyEvent::Rotation(e) => serialize_rotation_payload(e),
        KeyEvent::Interaction(e) => serialize_interaction_payload(e),
    }
}

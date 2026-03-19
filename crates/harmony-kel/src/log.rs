use alloc::vec::Vec;
use harmony_crypto::hash::{blake3_hash, truncated_hash};
use harmony_crypto::ml_dsa::{self, MlDsaPublicKey};
use harmony_crypto::ml_kem::MlKemPublicKey;
use harmony_identity::{CryptoSuite, IdentityHash, IdentityRef};
use serde::{Deserialize, Serialize};

use crate::commitment::verify_commitment;
use crate::error::KelError;
use crate::event::{
    serialize_event_payload, serialize_inception_payload, serialize_interaction_payload,
    serialize_rotation_payload, InceptionEvent, InteractionEvent, KeyEvent, RotationEvent,
};

const FORMAT_VERSION: u8 = 1;

/// Wire format for serializing/deserializing a `KeyEventLog`.
///
/// Only stores `address` and `events`; cached keys/commitment are re-derived
/// from the event chain on deserialization to prevent tampering.
#[derive(Serialize, Deserialize)]
struct KeyEventLogWire {
    address: IdentityHash,
    events: Vec<KeyEvent>,
}

/// Derive cached state (current signing key, encryption key, next-key commitment)
/// by scanning the event chain.
///
/// Panics if `events` is empty or `events[0]` is not an `InceptionEvent`.
fn derive_cached_state(events: &[KeyEvent]) -> (MlDsaPublicKey, MlKemPublicKey, [u8; 32]) {
    let mut signing_key;
    let mut encryption_key;
    let mut commitment;

    match &events[0] {
        KeyEvent::Inception(e) => {
            signing_key = e.signing_key.clone();
            encryption_key = e.encryption_key.clone();
            commitment = e.next_key_commitment;
        }
        _ => panic!("first event must be inception"),
    }

    for event in &events[1..] {
        if let KeyEvent::Rotation(e) = event {
            signing_key = e.signing_key.clone();
            encryption_key = e.encryption_key.clone();
            commitment = e.next_key_commitment;
        }
    }

    (signing_key, encryption_key, commitment)
}

/// A KERI-inspired Key Event Log with pre-rotation and dual signatures.
///
/// The log is append-only. The first event is always an inception, followed
/// by zero or more rotation and interaction events. Each event is chained to
/// the previous via a BLAKE3 hash of the serialized payload.
///
/// Current keys and commitment are cached for O(1) access.
#[derive(Debug, Clone)]
pub struct KeyEventLog {
    /// The permanent 128-bit address, derived from the inception payload.
    address: IdentityHash,
    /// Ordered list of all events in the log.
    events: Vec<KeyEvent>,
    /// Current active signing key (updated on rotation).
    current_signing_key: MlDsaPublicKey,
    /// Current active encryption key (updated on rotation).
    current_encryption_key: MlKemPublicKey,
    /// Commitment to the next key pair (updated on rotation).
    next_key_commitment: [u8; 32],
}

impl KeyEventLog {
    /// Create a new KEL from a verified inception event.
    ///
    /// Verifies the ML-DSA-65 signature over the unsigned payload and derives
    /// the permanent address via `truncated_hash(payload)`.
    pub fn from_inception(event: InceptionEvent) -> Result<Self, KelError> {
        // Compute unsigned payload for verification
        let payload = serialize_inception_payload(&event);

        // Verify the signature
        ml_dsa::verify(&event.signing_key, &payload, &event.signature)
            .map_err(|_| KelError::InvalidInceptionSignature)?;

        // Derive the permanent address from the unsigned payload
        let address = truncated_hash(&payload);

        let current_signing_key = event.signing_key.clone();
        let current_encryption_key = event.encryption_key.clone();
        let next_key_commitment = event.next_key_commitment;

        Ok(Self {
            address,
            events: alloc::vec![KeyEvent::Inception(event)],
            current_signing_key,
            current_encryption_key,
            next_key_commitment,
        })
    }

    /// Apply a rotation event to the log.
    ///
    /// Verifies:
    /// 1. Hash chain: `previous_hash == blake3(serialize(last_event))`
    /// 2. Pre-rotation commitment: new keys match the stored commitment
    /// 3. Sequence: must be exactly `latest_sequence + 1`
    /// 4. Old signature: signed by the *current* signing key
    /// 5. New signature: signed by the *rotation's* signing key
    pub fn apply_rotation(&mut self, event: RotationEvent) -> Result<(), KelError> {
        let last_event = self.events.last().ok_or(KelError::EmptyLog)?;

        // 1. Hash chain
        let expected_hash = blake3_hash(&serialize_event_payload(last_event));
        if event.previous_hash != expected_hash {
            return Err(KelError::HashChainBroken);
        }

        // 2. Pre-rotation commitment
        if !verify_commitment(
            &event.signing_key,
            &event.encryption_key,
            &self.next_key_commitment,
        ) {
            return Err(KelError::PreRotationMismatch);
        }

        // 3. Sequence
        let expected_seq = last_event.sequence() + 1;
        if event.sequence != expected_seq {
            return Err(KelError::SequenceViolation);
        }

        // 4. Old signature (current key signs the rotation payload)
        let payload = serialize_rotation_payload(&event);
        ml_dsa::verify(&self.current_signing_key, &payload, &event.old_signature)
            .map_err(|_| KelError::InvalidSignature)?;

        // 5. New signature (new key signs the rotation payload)
        ml_dsa::verify(&event.signing_key, &payload, &event.new_signature)
            .map_err(|_| KelError::InvalidSignature)?;

        // Update cached state
        self.current_signing_key = event.signing_key.clone();
        self.current_encryption_key = event.encryption_key.clone();
        self.next_key_commitment = event.next_key_commitment;

        self.events.push(KeyEvent::Rotation(event));
        Ok(())
    }

    /// Apply an interaction event to the log.
    ///
    /// Verifies:
    /// 1. Hash chain
    /// 2. Sequence
    /// 3. Signature under current signing key
    pub fn apply_interaction(&mut self, event: InteractionEvent) -> Result<(), KelError> {
        let last_event = self.events.last().ok_or(KelError::EmptyLog)?;

        // 1. Hash chain
        let expected_hash = blake3_hash(&serialize_event_payload(last_event));
        if event.previous_hash != expected_hash {
            return Err(KelError::HashChainBroken);
        }

        // 2. Sequence
        let expected_seq = last_event.sequence() + 1;
        if event.sequence != expected_seq {
            return Err(KelError::SequenceViolation);
        }

        // 3. Signature
        let payload = serialize_interaction_payload(&event);
        ml_dsa::verify(&self.current_signing_key, &payload, &event.signature)
            .map_err(|_| KelError::InvalidSignature)?;

        self.events.push(KeyEvent::Interaction(event));
        Ok(())
    }

    /// The permanent 128-bit address of this identity.
    pub fn address(&self) -> &IdentityHash {
        &self.address
    }

    /// The current active signing key.
    pub fn current_signing_key(&self) -> &MlDsaPublicKey {
        &self.current_signing_key
    }

    /// The current active encryption key.
    pub fn current_encryption_key(&self) -> &MlKemPublicKey {
        &self.current_encryption_key
    }

    /// The commitment to the next key pair.
    pub fn next_key_commitment(&self) -> &[u8; 32] {
        &self.next_key_commitment
    }

    /// Number of events in the log.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log is empty (should never be true for a valid log).
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// The sequence number of the latest event.
    pub fn latest_sequence(&self) -> u64 {
        self.events.last().map(|e| e.sequence()).unwrap_or(0)
    }

    /// Get an `IdentityRef` for this KEL identity.
    pub fn identity_ref(&self) -> IdentityRef {
        IdentityRef::new(self.address, CryptoSuite::MlDsa65Rotatable)
    }

    /// Serialize the log to bytes with a format version prefix.
    ///
    /// Only the address and events are serialized; cached keys/commitment
    /// are re-derived on deserialization.
    pub fn serialize(&self) -> Result<Vec<u8>, KelError> {
        let wire = KeyEventLogWire {
            address: self.address,
            events: self.events.clone(),
        };
        let mut buf = Vec::new();
        buf.push(FORMAT_VERSION);
        let inner = postcard::to_allocvec(&wire)
            .map_err(|_| KelError::SerializeError("postcard encode failed"))?;
        buf.extend_from_slice(&inner);
        Ok(buf)
    }

    /// Deserialize a log from bytes (expects format version prefix).
    ///
    /// Cached state (current signing key, encryption key, next-key commitment)
    /// is re-derived from the event chain rather than trusted from the wire.
    /// Validates that only `events[0]` is an inception event.
    pub fn deserialize(data: &[u8]) -> Result<Self, KelError> {
        if data.is_empty() {
            return Err(KelError::DeserializeError("empty data"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(KelError::DeserializeError("unsupported format version"));
        }
        let wire: KeyEventLogWire = postcard::from_bytes(&data[1..])
            .map_err(|_| KelError::DeserializeError("postcard decode failed"))?;

        if wire.events.is_empty() {
            return Err(KelError::EmptyLog);
        }

        // Validate: events[0] must be Inception, no other event may be Inception.
        if !matches!(wire.events[0], KeyEvent::Inception(_)) {
            return Err(KelError::DeserializeError("first event is not inception"));
        }
        for event in &wire.events[1..] {
            if matches!(event, KeyEvent::Inception(_)) {
                return Err(KelError::DuplicateInception);
            }
        }

        let (current_signing_key, current_encryption_key, next_key_commitment) =
            derive_cached_state(&wire.events);

        Ok(Self {
            address: wire.address,
            events: wire.events,
            current_signing_key,
            current_encryption_key,
            next_key_commitment,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{
        serialize_inception_payload_parts, serialize_interaction_payload_parts,
        serialize_rotation_payload_parts,
    };
    use harmony_crypto::{ml_dsa, ml_kem};
    use rand::rngs::OsRng;

    /// Helper struct holding a signing keypair and encryption public key.
    struct KeySet {
        sign_pk: MlDsaPublicKey,
        sign_sk: ml_dsa::MlDsaSecretKey,
        enc_pk: MlKemPublicKey,
    }

    fn gen_keyset() -> KeySet {
        let (sign_pk, sign_sk) = ml_dsa::generate(&mut OsRng);
        let (enc_pk, _enc_sk) = ml_kem::generate(&mut OsRng);
        KeySet {
            sign_pk,
            sign_sk,
            enc_pk,
        }
    }

    fn build_inception(keys: &KeySet, next: &KeySet) -> InceptionEvent {
        let commitment = crate::commitment::compute_commitment(&next.sign_pk, &next.enc_pk);
        let payload =
            serialize_inception_payload_parts(&keys.sign_pk, &keys.enc_pk, &commitment, 1000);
        let signature = ml_dsa::sign(&keys.sign_sk, &payload).unwrap();
        InceptionEvent {
            signing_key: keys.sign_pk.clone(),
            encryption_key: keys.enc_pk.clone(),
            next_key_commitment: commitment,
            created_at: 1000,
            signature,
        }
    }

    fn build_rotation(
        log: &KeyEventLog,
        old_keys: &KeySet,
        new_keys: &KeySet,
        next_keys: &KeySet,
    ) -> RotationEvent {
        let commitment =
            crate::commitment::compute_commitment(&next_keys.sign_pk, &next_keys.enc_pk);
        let last_event = &log.events[log.events.len() - 1];
        let previous_hash = blake3_hash(&serialize_event_payload(last_event));
        let sequence = last_event.sequence() + 1;

        let payload = serialize_rotation_payload_parts(
            sequence,
            &previous_hash,
            &new_keys.sign_pk,
            &new_keys.enc_pk,
            &commitment,
            2000 + sequence * 1000,
        );

        let old_signature = ml_dsa::sign(&old_keys.sign_sk, &payload).unwrap();
        let new_signature = ml_dsa::sign(&new_keys.sign_sk, &payload).unwrap();

        RotationEvent {
            sequence,
            previous_hash,
            signing_key: new_keys.sign_pk.clone(),
            encryption_key: new_keys.enc_pk.clone(),
            next_key_commitment: commitment,
            created_at: 2000 + sequence * 1000,
            old_signature,
            new_signature,
        }
    }

    fn build_interaction(
        log: &KeyEventLog,
        current_keys: &KeySet,
        data: &[u8],
    ) -> InteractionEvent {
        let last_event = &log.events[log.events.len() - 1];
        let previous_hash = blake3_hash(&serialize_event_payload(last_event));
        let sequence = last_event.sequence() + 1;
        let data_hash = blake3_hash(data);

        let payload = serialize_interaction_payload_parts(
            sequence,
            &previous_hash,
            &data_hash,
            3000 + sequence * 1000,
        );

        let signature = ml_dsa::sign(&current_keys.sign_sk, &payload).unwrap();

        InteractionEvent {
            sequence,
            previous_hash,
            data_hash,
            created_at: 3000 + sequence * 1000,
            signature,
        }
    }

    // ---- Test 1: from_inception_creates_valid_kel ----
    #[test]
    fn from_inception_creates_valid_kel() {
        let keys = gen_keyset();
        let next = gen_keyset();
        let inception = build_inception(&keys, &next);

        let log = KeyEventLog::from_inception(inception).unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log.latest_sequence(), 0);
        assert_eq!(
            log.current_signing_key().as_bytes(),
            keys.sign_pk.as_bytes()
        );
        assert_eq!(log.address().len(), 16);
    }

    // ---- Test 2: from_inception_rejects_invalid_signature ----
    #[test]
    fn from_inception_rejects_invalid_signature() {
        let keys = gen_keyset();
        let next = gen_keyset();
        let mut inception = build_inception(&keys, &next);
        // Tamper with created_at after signing
        inception.created_at = 9999;

        let result = KeyEventLog::from_inception(inception);
        assert_eq!(result.unwrap_err(), KelError::InvalidInceptionSignature);
    }

    // ---- Test 3: address_is_deterministic ----
    #[test]
    fn address_is_deterministic() {
        let keys = gen_keyset();
        let next = gen_keyset();

        // Build the same inception payload twice — address is derived from
        // the unsigned payload, not the signature, so different signatures
        // still produce the same address.
        let inception1 = build_inception(&keys, &next);
        let inception2 = build_inception(&keys, &next);

        let log1 = KeyEventLog::from_inception(inception1).unwrap();
        let log2 = KeyEventLog::from_inception(inception2).unwrap();

        assert_eq!(log1.address(), log2.address());
    }

    // ---- Test 4: apply_rotation_updates_keys ----
    #[test]
    fn apply_rotation_updates_keys() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();
        let keys2 = gen_keyset();

        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();

        let rotation = build_rotation(&log, &keys0, &keys1, &keys2);
        log.apply_rotation(rotation).unwrap();

        assert_eq!(
            log.current_signing_key().as_bytes(),
            keys1.sign_pk.as_bytes()
        );
        assert_eq!(log.len(), 2);
    }

    // ---- Test 5: apply_rotation_rejects_bad_commitment ----
    #[test]
    fn apply_rotation_rejects_bad_commitment() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();
        let keys2 = gen_keyset();
        let wrong_keys = gen_keyset(); // keys that don't match commitment

        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();

        // Build rotation with wrong_keys instead of keys1 (commitment is for keys1)
        let last_event = &log.events[log.events.len() - 1];
        let previous_hash = blake3_hash(&serialize_event_payload(last_event));
        let commitment = crate::commitment::compute_commitment(&keys2.sign_pk, &keys2.enc_pk);
        let payload = serialize_rotation_payload_parts(
            1,
            &previous_hash,
            &wrong_keys.sign_pk,
            &wrong_keys.enc_pk,
            &commitment,
            3000,
        );
        let old_signature = ml_dsa::sign(&keys0.sign_sk, &payload).unwrap();
        let new_signature = ml_dsa::sign(&wrong_keys.sign_sk, &payload).unwrap();

        let rotation = RotationEvent {
            sequence: 1,
            previous_hash,
            signing_key: wrong_keys.sign_pk.clone(),
            encryption_key: wrong_keys.enc_pk.clone(),
            next_key_commitment: commitment,
            created_at: 3000,
            old_signature,
            new_signature,
        };

        let result = log.apply_rotation(rotation);
        assert_eq!(result.unwrap_err(), KelError::PreRotationMismatch);
    }

    // ---- Test 6: apply_rotation_rejects_bad_hash_chain ----
    #[test]
    fn apply_rotation_rejects_bad_hash_chain() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();
        let keys2 = gen_keyset();

        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();

        let mut rotation = build_rotation(&log, &keys0, &keys1, &keys2);
        rotation.previous_hash = [0xAA; 32]; // wrong hash

        let result = log.apply_rotation(rotation);
        assert_eq!(result.unwrap_err(), KelError::HashChainBroken);
    }

    // ---- Test 7: apply_rotation_rejects_bad_sequence ----
    #[test]
    fn apply_rotation_rejects_bad_sequence() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();
        let keys2 = gen_keyset();

        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();

        // Build a valid rotation but change the sequence
        let last_event = &log.events[log.events.len() - 1];
        let previous_hash = blake3_hash(&serialize_event_payload(last_event));
        let commitment = crate::commitment::compute_commitment(&keys2.sign_pk, &keys2.enc_pk);
        let payload = serialize_rotation_payload_parts(
            5, // wrong sequence
            &previous_hash,
            &keys1.sign_pk,
            &keys1.enc_pk,
            &commitment,
            3000,
        );
        let old_signature = ml_dsa::sign(&keys0.sign_sk, &payload).unwrap();
        let new_signature = ml_dsa::sign(&keys1.sign_sk, &payload).unwrap();

        let rotation = RotationEvent {
            sequence: 5,
            previous_hash,
            signing_key: keys1.sign_pk.clone(),
            encryption_key: keys1.enc_pk.clone(),
            next_key_commitment: commitment,
            created_at: 3000,
            old_signature,
            new_signature,
        };

        let result = log.apply_rotation(rotation);
        assert_eq!(result.unwrap_err(), KelError::SequenceViolation);
    }

    // ---- Test 8: apply_rotation_rejects_bad_old_signature ----
    #[test]
    fn apply_rotation_rejects_bad_old_signature() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();
        let keys2 = gen_keyset();
        let wrong_keys = gen_keyset();

        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();

        // Build rotation where old_signature is from wrong key
        let last_event = &log.events[log.events.len() - 1];
        let previous_hash = blake3_hash(&serialize_event_payload(last_event));
        let commitment = crate::commitment::compute_commitment(&keys2.sign_pk, &keys2.enc_pk);
        let payload = serialize_rotation_payload_parts(
            1,
            &previous_hash,
            &keys1.sign_pk,
            &keys1.enc_pk,
            &commitment,
            3000,
        );
        let old_signature = ml_dsa::sign(&wrong_keys.sign_sk, &payload).unwrap(); // wrong key
        let new_signature = ml_dsa::sign(&keys1.sign_sk, &payload).unwrap();

        let rotation = RotationEvent {
            sequence: 1,
            previous_hash,
            signing_key: keys1.sign_pk.clone(),
            encryption_key: keys1.enc_pk.clone(),
            next_key_commitment: commitment,
            created_at: 3000,
            old_signature,
            new_signature,
        };

        let result = log.apply_rotation(rotation);
        assert_eq!(result.unwrap_err(), KelError::InvalidSignature);
    }

    // ---- Test 9: apply_rotation_rejects_bad_new_signature ----
    #[test]
    fn apply_rotation_rejects_bad_new_signature() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();
        let keys2 = gen_keyset();
        let wrong_keys = gen_keyset();

        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();

        // Build rotation where new_signature is from wrong key
        let last_event = &log.events[log.events.len() - 1];
        let previous_hash = blake3_hash(&serialize_event_payload(last_event));
        let commitment = crate::commitment::compute_commitment(&keys2.sign_pk, &keys2.enc_pk);
        let payload = serialize_rotation_payload_parts(
            1,
            &previous_hash,
            &keys1.sign_pk,
            &keys1.enc_pk,
            &commitment,
            3000,
        );
        let old_signature = ml_dsa::sign(&keys0.sign_sk, &payload).unwrap();
        let new_signature = ml_dsa::sign(&wrong_keys.sign_sk, &payload).unwrap(); // wrong key

        let rotation = RotationEvent {
            sequence: 1,
            previous_hash,
            signing_key: keys1.sign_pk.clone(),
            encryption_key: keys1.enc_pk.clone(),
            next_key_commitment: commitment,
            created_at: 3000,
            old_signature,
            new_signature,
        };

        let result = log.apply_rotation(rotation);
        assert_eq!(result.unwrap_err(), KelError::InvalidSignature);
    }

    // ---- Test 10: apply_interaction_appends ----
    #[test]
    fn apply_interaction_appends() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();

        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();
        assert_eq!(log.len(), 1);

        let interaction = build_interaction(&log, &keys0, b"hello world");
        log.apply_interaction(interaction).unwrap();

        assert_eq!(log.len(), 2);
        assert_eq!(log.latest_sequence(), 1);
    }

    // ---- Test 11: apply_interaction_rejects_bad_chain ----
    #[test]
    fn apply_interaction_rejects_bad_chain() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();

        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();

        let mut interaction = build_interaction(&log, &keys0, b"hello world");
        interaction.previous_hash = [0xBB; 32]; // wrong hash

        let result = log.apply_interaction(interaction);
        assert_eq!(result.unwrap_err(), KelError::HashChainBroken);
    }

    // ---- Test 12: apply_interaction_rejects_bad_sequence ----
    #[test]
    fn apply_interaction_rejects_bad_sequence() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();

        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();

        let last_event = &log.events[log.events.len() - 1];
        let previous_hash = blake3_hash(&serialize_event_payload(last_event));
        let data_hash = blake3_hash(b"test data");
        let payload = serialize_interaction_payload_parts(
            99, // wrong sequence
            &previous_hash,
            &data_hash,
            4000,
        );
        let signature = ml_dsa::sign(&keys0.sign_sk, &payload).unwrap();

        let interaction = InteractionEvent {
            sequence: 99,
            previous_hash,
            data_hash,
            created_at: 4000,
            signature,
        };

        let result = log.apply_interaction(interaction);
        assert_eq!(result.unwrap_err(), KelError::SequenceViolation);
    }

    // ---- Test 13: apply_interaction_rejects_bad_signature ----
    #[test]
    fn apply_interaction_rejects_bad_signature() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();
        let wrong_keys = gen_keyset();

        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();

        // Build interaction signed with wrong key
        let last_event = &log.events[log.events.len() - 1];
        let previous_hash = blake3_hash(&serialize_event_payload(last_event));
        let data_hash = blake3_hash(b"test data");
        let payload = serialize_interaction_payload_parts(1, &previous_hash, &data_hash, 4000);
        let signature = ml_dsa::sign(&wrong_keys.sign_sk, &payload).unwrap();

        let interaction = InteractionEvent {
            sequence: 1,
            previous_hash,
            data_hash,
            created_at: 4000,
            signature,
        };

        let result = log.apply_interaction(interaction);
        assert_eq!(result.unwrap_err(), KelError::InvalidSignature);
    }

    // ---- Test 14: address_permanent_through_rotations ----
    #[test]
    fn address_permanent_through_rotations() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();
        let keys2 = gen_keyset();

        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();
        let original_address = *log.address();

        let rotation = build_rotation(&log, &keys0, &keys1, &keys2);
        log.apply_rotation(rotation).unwrap();

        assert_eq!(*log.address(), original_address);
    }

    // ---- Test 15: identity_ref_returns_rotatable ----
    #[test]
    fn identity_ref_returns_rotatable() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();

        let inception = build_inception(&keys0, &keys1);
        let log = KeyEventLog::from_inception(inception).unwrap();
        let id_ref = log.identity_ref();

        assert_eq!(id_ref.suite, CryptoSuite::MlDsa65Rotatable);
        assert_eq!(id_ref.hash, *log.address());
        assert!(id_ref.is_post_quantum());
    }

    // ---- Test 16: double_rotation ----
    #[test]
    fn double_rotation() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();
        let keys2 = gen_keyset();
        let keys3 = gen_keyset();

        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();

        let rot1 = build_rotation(&log, &keys0, &keys1, &keys2);
        log.apply_rotation(rot1).unwrap();

        let rot2 = build_rotation(&log, &keys1, &keys2, &keys3);
        log.apply_rotation(rot2).unwrap();

        assert_eq!(log.len(), 3);
        assert_eq!(log.latest_sequence(), 2);
        assert_eq!(
            log.current_signing_key().as_bytes(),
            keys2.sign_pk.as_bytes()
        );
    }

    // ---- Test 17: interaction_after_rotation ----
    #[test]
    fn interaction_after_rotation() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();
        let keys2 = gen_keyset();

        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();

        let rotation = build_rotation(&log, &keys0, &keys1, &keys2);
        log.apply_rotation(rotation).unwrap();

        // Interaction must be signed with the new current key (keys1)
        let interaction = build_interaction(&log, &keys1, b"post-rotation data");
        log.apply_interaction(interaction).unwrap();

        assert_eq!(log.len(), 3);
        assert_eq!(log.latest_sequence(), 2);
    }

    // ---- Test 18: mixed_event_sequence ----
    #[test]
    fn mixed_event_sequence() {
        let keys0 = gen_keyset();
        let keys1 = gen_keyset();
        let keys2 = gen_keyset();

        // inception → interaction → rotation → interaction
        let inception = build_inception(&keys0, &keys1);
        let mut log = KeyEventLog::from_inception(inception).unwrap();

        let ixn1 = build_interaction(&log, &keys0, b"first interaction");
        log.apply_interaction(ixn1).unwrap();

        let rot = build_rotation(&log, &keys0, &keys1, &keys2);
        log.apply_rotation(rot).unwrap();

        let ixn2 = build_interaction(&log, &keys1, b"second interaction");
        log.apply_interaction(ixn2).unwrap();

        assert_eq!(log.len(), 4);
        assert_eq!(log.latest_sequence(), 3);
        assert_eq!(
            log.current_signing_key().as_bytes(),
            keys1.sign_pk.as_bytes()
        );
    }

    // ---- Test 19: serialize_deserialize_round_trip ----
    // ML-DSA/ML-KEM types are very large; postcard deserialization of a full
    // KeyEventLog can overflow the default 2 MiB test-thread stack, so we
    // spawn a thread with an explicit 8 MiB stack.
    #[test]
    fn serialize_deserialize_round_trip() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let keys0 = gen_keyset();
                let keys1 = gen_keyset();
                let keys2 = gen_keyset();

                let inception = build_inception(&keys0, &keys1);
                let mut log = KeyEventLog::from_inception(inception).unwrap();

                let rotation = build_rotation(&log, &keys0, &keys1, &keys2);
                log.apply_rotation(rotation).unwrap();

                let ixn = build_interaction(&log, &keys1, b"persistence test");
                log.apply_interaction(ixn).unwrap();

                // Serialize and deserialize
                let bytes = log.serialize().unwrap();
                let restored = KeyEventLog::deserialize(&bytes).unwrap();

                assert_eq!(restored.address(), log.address());
                assert_eq!(restored.len(), log.len());
                assert_eq!(restored.latest_sequence(), log.latest_sequence());
                assert_eq!(
                    restored.current_signing_key().as_bytes(),
                    log.current_signing_key().as_bytes()
                );
                assert_eq!(
                    restored.current_encryption_key().as_bytes(),
                    log.current_encryption_key().as_bytes()
                );
                assert_eq!(restored.next_key_commitment(), log.next_key_commitment());
            })
            .expect("failed to spawn thread")
            .join()
            .expect("thread panicked");
    }

    // ---- Test 20: deserialize_rejects_corrupt_data ----
    #[test]
    fn deserialize_rejects_corrupt_data() {
        // Empty data
        assert!(matches!(
            KeyEventLog::deserialize(&[]),
            Err(KelError::DeserializeError("empty data"))
        ));

        // Wrong version
        assert!(matches!(
            KeyEventLog::deserialize(&[0xFF, 0x01, 0x02]),
            Err(KelError::DeserializeError("unsupported format version"))
        ));

        // Correct version but garbage payload
        assert!(matches!(
            KeyEventLog::deserialize(&[FORMAT_VERSION, 0xFF, 0xFF, 0xFF]),
            Err(KelError::DeserializeError("postcard decode failed"))
        ));
    }
}

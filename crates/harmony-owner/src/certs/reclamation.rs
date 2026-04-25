use crate::cbor;
use crate::pubkey_bundle::PubKeyBundle;
use crate::signing::{sign_with_tag, tags, verify_with_tag};
use crate::OwnerError;
use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

pub const RECLAMATION_VERSION: u8 = 1;

/// Default challenge window for reclamation: 30 days in seconds.
pub const DEFAULT_CHALLENGE_WINDOW_SECS: u64 = 30 * 24 * 60 * 60;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReclamationCert {
    pub version: u8,
    pub new_owner_id: [u8; 16],
    pub new_owner_pubkey: PubKeyBundle,
    pub claimed_predecessor: [u8; 16],
    pub issued_at: u64,
    pub challenge_window_end: u64,
    pub note: String,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
struct ReclamationSigningPayload<'a> {
    version: u8,
    new_owner_id: [u8; 16],
    new_owner_pubkey: &'a PubKeyBundle,
    claimed_predecessor: [u8; 16],
    issued_at: u64,
    challenge_window_end: u64,
    note: &'a str,
}

impl ReclamationCert {
    pub fn sign(
        new_master_sk: &SigningKey,
        new_owner_pubkey: PubKeyBundle,
        claimed_predecessor: [u8; 16],
        issued_at: u64,
        challenge_window_secs: u64,
        note: String,
    ) -> Result<Self, OwnerError> {
        let new_owner_id = new_owner_pubkey.identity_hash();
        if challenge_window_secs == 0 {
            return Err(OwnerError::InvalidChallengeWindow);
        }
        let challenge_window_end = issued_at
            .checked_add(challenge_window_secs)
            .ok_or(OwnerError::InvalidChallengeWindow)?;
        let payload_bytes = cbor::to_canonical(&ReclamationSigningPayload {
            version: RECLAMATION_VERSION,
            new_owner_id,
            new_owner_pubkey: &new_owner_pubkey,
            claimed_predecessor,
            issued_at,
            challenge_window_end,
            note: &note,
        })?;
        let signature = sign_with_tag(new_master_sk, tags::RECLAMATION, &payload_bytes);
        Ok(ReclamationCert {
            version: RECLAMATION_VERSION,
            new_owner_id,
            new_owner_pubkey,
            claimed_predecessor,
            issued_at,
            challenge_window_end,
            note,
            signature,
        })
    }

    pub fn verify(&self) -> Result<(), OwnerError> {
        if self.version != RECLAMATION_VERSION {
            return Err(OwnerError::UnknownVersion(self.version));
        }
        if self.challenge_window_end <= self.issued_at {
            return Err(OwnerError::InvalidChallengeWindow);
        }
        if self.new_owner_pubkey.identity_hash() != self.new_owner_id {
            return Err(OwnerError::IdentityHashMismatch);
        }
        let vk = VerifyingKey::from_bytes(&self.new_owner_pubkey.classical.ed25519_verify)
            .map_err(|_| OwnerError::InvalidSignature { cert_type: "Reclamation" })?;
        let payload_bytes = cbor::to_canonical(&ReclamationSigningPayload {
            version: self.version,
            new_owner_id: self.new_owner_id,
            new_owner_pubkey: &self.new_owner_pubkey,
            claimed_predecessor: self.claimed_predecessor,
            issued_at: self.issued_at,
            challenge_window_end: self.challenge_window_end,
            note: &self.note,
        })?;
        verify_with_tag(&vk, tags::RECLAMATION, &payload_bytes, &self.signature, "Reclamation")
    }

    /// True if a `LivenessCert` with timestamp > self.issued_at and signed by
    /// any device under the predecessor identity refutes this reclamation.
    /// (The actual lookup of "any device under predecessor" is OwnerState's job.)
    pub fn is_refuted_by_timestamp(&self, liveness_timestamp: u64) -> bool {
        liveness_timestamp > self.issued_at && liveness_timestamp <= self.challenge_window_end
    }

    pub fn is_window_expired(&self, now: u64) -> bool {
        now >= self.challenge_window_end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pubkey_bundle::ClassicalKeys;
    use rand::rngs::OsRng;

    #[test]
    fn reclamation_verifies() {
        let sk = SigningKey::generate(&mut OsRng);
        let bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
            post_quantum: None,
        };
        let cert = ReclamationCert::sign(
            &sk, bundle, [9u8; 16], 1_700_000_000, DEFAULT_CHALLENGE_WINDOW_SECS, "lost devices in fire".into()
        ).unwrap();
        cert.verify().unwrap();
    }

    #[test]
    fn refutation_logic() {
        let sk = SigningKey::generate(&mut OsRng);
        let bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
            post_quantum: None,
        };
        let cert = ReclamationCert::sign(&sk, bundle, [9u8; 16], 1_000_000, 1000, "n/a".into()).unwrap();
        assert!(cert.is_refuted_by_timestamp(1_000_001));
        assert!(!cert.is_refuted_by_timestamp(999_999));
        assert!(!cert.is_refuted_by_timestamp(1_000_000));
        assert!(cert.is_window_expired(1_001_000));
        assert!(!cert.is_window_expired(1_000_500));
    }

    #[test]
    fn reclamation_with_inverted_window_rejected_in_verify() {
        let sk = SigningKey::generate(&mut OsRng);
        let bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
            post_quantum: None,
        };
        let mut cert = ReclamationCert::sign(
            &sk, bundle.clone(), [9u8; 16], 1_000_000, DEFAULT_CHALLENGE_WINDOW_SECS, "n/a".into()
        ).unwrap();
        // Manually corrupt window_end to be before issued_at
        cert.challenge_window_end = cert.issued_at - 1;
        let result = cert.verify();
        assert!(matches!(result, Err(OwnerError::InvalidChallengeWindow)));
    }

    #[test]
    fn zero_length_window_rejected_in_sign() {
        let sk = SigningKey::generate(&mut OsRng);
        let bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
            post_quantum: None,
        };
        let result = ReclamationCert::sign(&sk, bundle, [9u8; 16], 1_000_000, 0, "n/a".into());
        assert!(matches!(result, Err(OwnerError::InvalidChallengeWindow)));
    }

    #[test]
    fn zero_length_window_rejected_in_verify() {
        let sk = SigningKey::generate(&mut OsRng);
        let bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
            post_quantum: None,
        };
        let mut cert = ReclamationCert::sign(
            &sk, bundle, [9u8; 16], 1_000_000, DEFAULT_CHALLENGE_WINDOW_SECS, "n/a".into()
        ).unwrap();
        // Force end == start (zero-length on the wire)
        cert.challenge_window_end = cert.issued_at;
        let result = cert.verify();
        assert!(matches!(result, Err(OwnerError::InvalidChallengeWindow)));
    }

    #[test]
    fn refutation_outside_window_does_not_refute() {
        let sk = SigningKey::generate(&mut OsRng);
        let bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
            post_quantum: None,
        };
        let cert = ReclamationCert::sign(&sk, bundle, [9u8; 16], 1_000_000, 1000, "n/a".into()).unwrap();
        // Liveness AFTER window expires (window ends at 1_001_000)
        assert!(!cert.is_refuted_by_timestamp(1_002_000));
        // Liveness AT window end is still valid (boundary is inclusive)
        assert!(cert.is_refuted_by_timestamp(1_001_000));
    }
}

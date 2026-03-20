use alloc::vec::Vec;
use harmony_identity::IdentityRef;
use serde::{Deserialize, Serialize};

use crate::error::DiscoveryError;

const FORMAT_VERSION: u8 = 1;

/// Internal struct for the signable portion of an announce record.
/// Everything except the signature.
#[derive(Serialize, Deserialize)]
struct SignablePayload {
    format_version: u8,
    identity_ref: IdentityRef,
    public_key: Vec<u8>,
    routing_hints: Vec<RoutingHint>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

/// A routing hint that tells peers how to reach this identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingHint {
    /// Reticulum destination hash (16 bytes).
    Reticulum { destination_hash: [u8; 16] },
    /// Zenoh locator (e.g. `tcp/192.168.1.1:7447`).
    Zenoh { locator: Vec<u8> },
}

/// A signed announce record that advertises an identity's presence and
/// reachability on the network.
///
/// Produced by `AnnounceBuilder`. Verified by `AnnounceVerifier`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnounceRecord {
    pub identity_ref: IdentityRef,
    pub public_key: Vec<u8>,
    pub routing_hints: Vec<RoutingHint>,
    pub published_at: u64,
    pub expires_at: u64,
    pub nonce: [u8; 16],
    pub signature: Vec<u8>,
}

impl AnnounceRecord {
    /// Reconstruct the signable payload bytes (everything except signature).
    pub(crate) fn signable_bytes(&self) -> Vec<u8> {
        let payload = SignablePayload {
            format_version: FORMAT_VERSION,
            identity_ref: self.identity_ref,
            public_key: self.public_key.clone(),
            routing_hints: self.routing_hints.clone(),
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    /// Serialize the record to bytes with a format version prefix.
    pub fn serialize(&self) -> Result<Vec<u8>, DiscoveryError> {
        let mut buf = Vec::new();
        buf.push(FORMAT_VERSION);
        let inner = postcard::to_allocvec(self)
            .map_err(|_| DiscoveryError::SerializeError("postcard encode failed"))?;
        buf.extend_from_slice(&inner);
        Ok(buf)
    }

    /// Deserialize a record from bytes (expects format version prefix).
    pub fn deserialize(data: &[u8]) -> Result<Self, DiscoveryError> {
        if data.is_empty() {
            return Err(DiscoveryError::DeserializeError("empty data"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(DiscoveryError::DeserializeError(
                "unsupported format version",
            ));
        }
        postcard::from_bytes(&data[1..])
            .map_err(|_| DiscoveryError::DeserializeError("postcard decode failed"))
    }
}

/// Builder for constructing an announce record.
///
/// Sans-I/O: call `signable_payload()` to get the bytes to sign,
/// then `build(signature)` to produce the final record.
///
/// **Important:** Do not modify the builder between calling
/// `signable_payload()` and `build()`. The signature covers the payload
/// at the time `signable_payload()` was called; any subsequent mutation
/// will cause signature verification to fail.
pub struct AnnounceBuilder {
    identity_ref: IdentityRef,
    public_key: Vec<u8>,
    routing_hints: Vec<RoutingHint>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

impl AnnounceBuilder {
    /// Create a new builder.
    ///
    /// # Panics
    ///
    /// Panics if `expires_at <= published_at` (would produce a permanently
    /// invalid record).
    pub fn new(
        identity_ref: IdentityRef,
        public_key: Vec<u8>,
        published_at: u64,
        expires_at: u64,
        nonce: [u8; 16],
    ) -> Self {
        assert!(
            expires_at > published_at,
            "expires_at ({expires_at}) must be > published_at ({published_at})"
        );
        Self {
            identity_ref,
            public_key,
            routing_hints: Vec::new(),
            published_at,
            expires_at,
            nonce,
        }
    }

    /// Add a routing hint.
    pub fn add_routing_hint(&mut self, hint: RoutingHint) -> &mut Self {
        self.routing_hints.push(hint);
        self
    }

    /// Produce the signable payload bytes.
    ///
    /// The caller signs these bytes externally with their private key.
    pub fn signable_payload(&self) -> Vec<u8> {
        let payload = SignablePayload {
            format_version: FORMAT_VERSION,
            identity_ref: self.identity_ref,
            public_key: self.public_key.clone(),
            routing_hints: self.routing_hints.clone(),
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    /// Finalize with a signature to produce the `AnnounceRecord`.
    pub fn build(self, signature: Vec<u8>) -> AnnounceRecord {
        AnnounceRecord {
            identity_ref: self.identity_ref,
            public_key: self.public_key,
            routing_hints: self.routing_hints,
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
            signature,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_identity::{CryptoSuite, IdentityRef};

    fn test_identity() -> IdentityRef {
        IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519)
    }

    fn test_public_key() -> Vec<u8> {
        alloc::vec![0x01, 0x02, 0x03, 0x04]
    }

    #[test]
    fn builder_produces_correct_fields() {
        let hint = RoutingHint::Reticulum {
            destination_hash: [0xDE; 16],
        };
        let mut builder = AnnounceBuilder::new(
            test_identity(),
            test_public_key(),
            1000,
            2000,
            [0x01; 16],
        );
        builder.add_routing_hint(hint.clone());
        let record = builder.build(alloc::vec![0x51, 0x67]);

        assert_eq!(record.identity_ref, test_identity());
        assert_eq!(record.public_key, test_public_key());
        assert_eq!(record.published_at, 1000);
        assert_eq!(record.expires_at, 2000);
        assert_eq!(record.nonce, [0x01; 16]);
        assert_eq!(record.routing_hints.len(), 1);
        assert_eq!(record.routing_hints[0], hint);
    }

    #[test]
    fn signable_payload_is_deterministic() {
        let builder = AnnounceBuilder::new(
            test_identity(),
            test_public_key(),
            1000,
            2000,
            [0x01; 16],
        );
        let p1 = builder.signable_payload();
        let p2 = builder.signable_payload();
        assert_eq!(p1, p2);
    }

    #[test]
    #[should_panic(expected = "expires_at")]
    fn rejects_expires_at_before_published_at() {
        AnnounceBuilder::new(test_identity(), test_public_key(), 2000, 1000, [0x01; 16]);
    }

    #[test]
    #[should_panic(expected = "expires_at")]
    fn rejects_expires_at_equal_to_published_at() {
        AnnounceBuilder::new(test_identity(), test_public_key(), 1000, 1000, [0x01; 16]);
    }

    #[test]
    fn multiple_routing_hints() {
        let mut builder = AnnounceBuilder::new(
            test_identity(),
            test_public_key(),
            1000,
            2000,
            [0x01; 16],
        );
        builder.add_routing_hint(RoutingHint::Reticulum {
            destination_hash: [0xAA; 16],
        });
        builder.add_routing_hint(RoutingHint::Zenoh {
            locator: alloc::vec![b't', b'c', b'p'],
        });
        let record = builder.build(alloc::vec![]);
        assert_eq!(record.routing_hints.len(), 2);
    }

    #[test]
    fn serde_round_trip() {
        let mut builder = AnnounceBuilder::new(
            test_identity(),
            test_public_key(),
            1000,
            2000,
            [0x01; 16],
        );
        builder.add_routing_hint(RoutingHint::Reticulum {
            destination_hash: [0xBB; 16],
        });
        let record = builder.build(alloc::vec![0xDE, 0xAD]);

        let bytes = record.serialize().unwrap();
        let restored = AnnounceRecord::deserialize(&bytes).unwrap();

        assert_eq!(restored.identity_ref, record.identity_ref);
        assert_eq!(restored.public_key, record.public_key);
        assert_eq!(restored.routing_hints, record.routing_hints);
        assert_eq!(restored.published_at, record.published_at);
        assert_eq!(restored.expires_at, record.expires_at);
        assert_eq!(restored.nonce, record.nonce);
        assert_eq!(restored.signature, record.signature);
    }

    #[test]
    fn deserialize_rejects_corrupt_data() {
        assert!(matches!(
            AnnounceRecord::deserialize(&[]),
            Err(DiscoveryError::DeserializeError("empty data"))
        ));
        assert!(matches!(
            AnnounceRecord::deserialize(&[0xFF]),
            Err(DiscoveryError::DeserializeError(
                "unsupported format version"
            ))
        ));
    }
}

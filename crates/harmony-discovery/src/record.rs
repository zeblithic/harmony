use alloc::vec::Vec;
use harmony_identity::IdentityRef;
use serde::{Deserialize, Serialize};

use crate::error::DiscoveryError;

const FORMAT_VERSION: u8 = 2;

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
    /// Zenoh locator (e.g. `"tcp/192.168.1.1:7447"`).
    Zenoh { locator: alloc::string::String },
    /// iroh-net tunnel: NodeId + optional relay + optional direct addresses.
    Tunnel {
        /// BLAKE3(ML-DSA-65 public key) — the iroh NodeId seed.
        node_id: [u8; 32],
        /// Preferred relay server URL (e.g., "https://iroh.q8.fyi").
        relay_url: Option<alloc::string::String>,
        /// Known direct socket addresses (ephemeral, may be stale).
        direct_addrs: alloc::vec::Vec<alloc::string::String>,
    },
}

/// A signed announce record that advertises an identity's presence and
/// reachability on the network.
///
/// # Construction
///
/// Produce records via [`AnnounceBuilder`]; direct struct construction
/// bypasses validity checks (e.g. `expires_at > published_at`) and may
/// cause downstream verification failures.
///
/// Verified by [`verify_announce()`](crate::verify::verify_announce).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnnounceRecord {
    pub identity_ref: IdentityRef,
    pub public_key: Vec<u8>,
    pub routing_hints: Vec<RoutingHint>,
    pub published_at: u64,
    pub expires_at: u64,
    /// Random bytes for replay protection. Must be cryptographically
    /// random and unique per announce — especially when `published_at`
    /// has not advanced since the last announcement.
    pub nonce: [u8; 16],
    pub signature: Vec<u8>,
}

impl AnnounceRecord {
    /// Reconstruct the signable payload bytes (everything except signature).
    ///
    /// Uses the compile-time `FORMAT_VERSION` constant. This assumes all
    /// `AnnounceRecord` instances in this process were created under the
    /// same version. Records from persistent storage MUST go through
    /// [`AnnounceRecord::deserialize`] (which gates on FORMAT_VERSION)
    /// rather than being passed as deserialized structs.
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
    ///
    /// The version byte is a hard gate: records serialized under one
    /// version cannot be deserialized by a binary compiled with a
    /// different `FORMAT_VERSION`. During rolling upgrades, nodes on
    /// the old version will reject new-version records and vice versa.
    /// This is intentional — a version bump signals a struct layout
    /// change that requires coordinated deployment.
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
        // Strict version check. V1 records would fail signature verification
        // because signable_bytes() uses the compile-time FORMAT_VERSION (2),
        // producing different payload bytes than the v1 signer used. Rejecting
        // v1 upfront gives a clear error instead of a cryptic signature failure.
        if data[0] != FORMAT_VERSION {
            return Err(DiscoveryError::DeserializeError(
                "unsupported announce format version",
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
    /// All timestamps (`published_at`, `expires_at`) are Unix epoch
    /// seconds. `nonce` must be 16 cryptographically random bytes,
    /// unique per announce — especially when `published_at` has not
    /// advanced since the last announcement.
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
        let mut builder =
            AnnounceBuilder::new(test_identity(), test_public_key(), 1000, 2000, [0x01; 16]);
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
        let builder =
            AnnounceBuilder::new(test_identity(), test_public_key(), 1000, 2000, [0x01; 16]);
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
        let mut builder =
            AnnounceBuilder::new(test_identity(), test_public_key(), 1000, 2000, [0x01; 16]);
        builder.add_routing_hint(RoutingHint::Reticulum {
            destination_hash: [0xAA; 16],
        });
        builder.add_routing_hint(RoutingHint::Zenoh {
            locator: alloc::string::String::from("tcp/127.0.0.1:7447"),
        });
        let record = builder.build(alloc::vec![]);
        assert_eq!(record.routing_hints.len(), 2);
    }

    #[test]
    fn serde_round_trip() {
        let mut builder =
            AnnounceBuilder::new(test_identity(), test_public_key(), 1000, 2000, [0x01; 16]);
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

    #[test]
    fn tunnel_routing_hint_roundtrip() {
        let hint = RoutingHint::Tunnel {
            node_id: [0xAA; 32],
            relay_url: Some("https://iroh.q8.fyi".into()),
            direct_addrs: vec!["192.168.1.10:4242".into()],
        };
        let bytes = postcard::to_allocvec(&hint).unwrap();
        let decoded: RoutingHint = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(hint, decoded);
    }

    #[test]
    fn announce_record_with_tunnel_hint_roundtrip() {
        let hint = RoutingHint::Tunnel {
            node_id: [0xBB; 32],
            relay_url: Some("https://relay.example.com".into()),
            direct_addrs: vec!["10.0.0.1:7777".into(), "203.0.113.5:7777".into()],
        };
        let mut builder =
            AnnounceBuilder::new(test_identity(), test_public_key(), 1000, 2000, [0x42; 16]);
        builder.add_routing_hint(hint.clone());
        let record = builder.build(alloc::vec![0xCA, 0xFE]);

        let bytes = record.serialize().unwrap();
        let restored = AnnounceRecord::deserialize(&bytes).unwrap();

        assert_eq!(restored.routing_hints.len(), 1);
        assert_eq!(restored.routing_hints[0], hint);
        assert_eq!(restored.identity_ref, record.identity_ref);
        assert_eq!(restored.published_at, record.published_at);
        assert_eq!(restored.expires_at, record.expires_at);
        assert_eq!(restored.nonce, record.nonce);
        assert_eq!(restored.signature, record.signature);
    }
}

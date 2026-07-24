//! Reachability announce record: the node-id / relay / direct-addrs / identity-sig
//! quad (+ an optional delegate-endpoint advertisement) published to the pkarr
//! DHT. Byte-stable CBOR — see the golden vector in tests.
//!
//! Moved from harmony-client `reachability_record.rs` (ZEB-744). The inner
//! identity signature and delegate/butler *policy* stay in the app; this crate
//! carries only the record shape + its byte-stable encoding.

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

use crate::canonical::{
    canonical_cbor_encode, deserialize_bytes_from_bstr, is_zero_u64, serialize_bytes_as_bstr,
    CborError,
};

/// One delegate endpoint advertised in a reachability record: another device of
/// the same owner that can be reached (and, in the app, can accept sealed
/// deposits) on the owner's behalf. Byte-identical to harmony-client's
/// `ButlerSetEntry` (renamed; wire keys `d`/`ep`/`vk`/`hr`/`pn` unchanged).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegateEndpoint {
    /// 16-byte identity hash of the device.
    #[serde(
        rename = "d",
        serialize_with = "serialize_bytes_as_bstr",
        deserialize_with = "deserialize_bytes_from_bstr"
    )]
    pub device_id: [u8; 16],

    /// Iroh EndpointId / NodeId (32-byte transport key).
    #[serde(
        rename = "ep",
        serialize_with = "serialize_bytes_as_bstr",
        deserialize_with = "deserialize_bytes_from_bstr"
    )]
    pub iroh_endpoint_id: [u8; 32],

    /// The device's Ed25519 verify key.
    #[serde(
        rename = "vk",
        serialize_with = "serialize_bytes_as_bstr",
        deserialize_with = "deserialize_bytes_from_bstr"
    )]
    pub device_ed25519_verify: [u8; 32],

    /// Home relay URL for dialing the device.
    #[serde(rename = "hr")]
    pub home_relay: String,

    /// Pinned always-on device.
    #[serde(rename = "pn")]
    pub pinned: bool,
}

/// A reachability announce record. Field order and `#[serde(rename)]` keys are
/// the byte-compat surface — do not reorder or rename. All keys are 2 chars to
/// keep the same-length-keys CBOR determinism invariant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReachabilityAnnouncePayload {
    /// Iroh NodeId (Ed25519 public key, 32 bytes).
    #[serde(
        rename = "nd",
        serialize_with = "serialize_bytes_as_bstr",
        deserialize_with = "deserialize_bytes_from_bstr"
    )]
    pub iroh_node_id: [u8; 32],

    /// Home relay URL.
    #[serde(rename = "rl")]
    pub home_relay_url: String,

    /// Direct-traversal hint addresses (may be empty).
    #[serde(rename = "da")]
    pub direct_addresses: Vec<SocketAddr>,

    /// Wall-clock milliseconds when this record was authored.
    #[serde(rename = "ts")]
    pub announced_at_ms: u64,

    /// Inner Ed25519 signature by the author's identity key. Computed and
    /// verified by the app (the preimage binds app-side envelope fields);
    /// zero-filled on the pkarr-published path. 64 bytes.
    #[serde(
        rename = "sg",
        serialize_with = "serialize_bytes_as_bstr",
        deserialize_with = "deserialize_bytes_from_bstr"
    )]
    pub identity_signature: [u8; 64],

    /// Ordered delegate-endpoint advertisement. Elided when empty so records
    /// without it stay byte-identical to the legacy encoding.
    #[serde(rename = "bs", default, skip_serializing_if = "Vec::is_empty")]
    pub butler_set: Vec<DelegateEndpoint>,

    /// Wall-clock ms freshness stamp for the delegate set. Zero (elided) means
    /// "no delegate set".
    #[serde(rename = "ba", default, skip_serializing_if = "is_zero_u64")]
    pub bs_at: u64,
}

/// Canonical-encode for hashing / signing / publishing.
pub fn canonical_payload_bytes(p: &ReachabilityAnnouncePayload) -> Result<Vec<u8>, CborError> {
    canonical_cbor_encode(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_payload() -> ReachabilityAnnouncePayload {
        ReachabilityAnnouncePayload {
            iroh_node_id: [0xAB; 32],
            home_relay_url: "https://derp.example/".into(),
            direct_addresses: vec![],
            announced_at_ms: 1_700_000_000_000,
            identity_signature: [0xCD; 64],
            butler_set: Vec::new(),
            bs_at: 0,
        }
    }

    /// Byte-compat lock (migrated from harmony-client, DO NOT REGENERATE): a
    /// record WITHOUT a butler set must encode byte-identically to the legacy
    /// (pre-butler-set) wire encoding, so already-deployed peers keep decoding
    /// published pkarr routing blobs unchanged.
    #[test]
    fn routing_blob_without_butler_set_is_wire_identical_to_legacy() {
        const EXPECTED_LEGACY_HEX: &str = "a5626e645820abababababababababababababababababababababababababababababababab62726c7568747470733a2f2f646572702e6578616d706c652f626461806274731b0000018bcfe568006273675840cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd";
        let p = fixture_payload();
        let bytes = canonical_payload_bytes(&p).expect("encode");
        assert_eq!(hex::encode(&bytes), EXPECTED_LEGACY_HEX, "legacy wire encoding drifted");
    }

    fn fixture_delegate(seed: u8) -> DelegateEndpoint {
        DelegateEndpoint {
            device_id: [seed; 16],
            iroh_endpoint_id: [seed.wrapping_add(1); 32],
            device_ed25519_verify: [seed.wrapping_add(2); 32],
            home_relay: "https://use1-1.relay.iroh.network./".into(),
            pinned: false,
        }
    }

    #[test]
    fn legacy_routing_blob_decodes_with_empty_butler_set() {
        let legacy = canonical_payload_bytes(&fixture_payload()).expect("encode");
        let decoded: ReachabilityAnnouncePayload =
            ciborium::de::from_reader(&legacy[..]).expect("decode legacy");
        assert!(decoded.butler_set.is_empty());
        assert_eq!(decoded.bs_at, 0);
    }

    #[test]
    fn routing_blob_with_butler_set_round_trips() {
        let mut p = fixture_payload();
        p.butler_set = vec![fixture_delegate(0x10), fixture_delegate(0x20)];
        p.bs_at = 1_700_000_000_000;
        let bytes = canonical_payload_bytes(&p).expect("encode");
        let decoded: ReachabilityAnnouncePayload =
            ciborium::de::from_reader(&bytes[..]).expect("decode");
        assert_eq!(decoded, p);
    }

    #[test]
    fn roundtrip_cbor() {
        let p = fixture_payload();
        let bytes = canonical_payload_bytes(&p).expect("encode");
        let decoded: ReachabilityAnnouncePayload =
            ciborium::de::from_reader(&bytes[..]).expect("decode");
        assert_eq!(decoded, p);
    }

    #[test]
    fn payload_keys_are_2_chars() {
        let bytes = canonical_payload_bytes(&fixture_payload()).expect("encode");
        let val: ciborium::Value = ciborium::de::from_reader(&bytes[..]).expect("decode");
        for (k, _) in val.as_map().expect("payload is map") {
            assert_eq!(k.as_text().expect("key is text").chars().count(), 2);
        }
    }

    #[test]
    fn encoded_size_with_two_entries_under_bep44_budget() {
        let p = ReachabilityAnnouncePayload {
            iroh_node_id: [0xAB; 32],
            home_relay_url: "https://use1-1.relay.iroh.network./".into(),
            direct_addresses: vec![
                "203.0.113.7:62103".parse().expect("v4"),
                "[2001:db8::1234:5678]:62103".parse().expect("v6"),
            ],
            announced_at_ms: 1_700_000_000_000,
            identity_signature: [0xCD; 64],
            butler_set: vec![fixture_delegate(0x10), fixture_delegate(0x20)],
            bs_at: 1_700_000_000_000,
        };
        assert!(canonical_payload_bytes(&p).expect("encode").len() < 900);
    }
}

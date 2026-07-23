//! Tunnel wire-protocol versioning primitives (ported from harmony-client
//! `protocol_versioning.rs`, ZEB-623) — the subset the tunnel driver needs.
//!
//! Two evolution mechanisms: an ALPN *generation* bump (rare, wire-incompatible;
//! [`alpn::HARMONY_TUNNEL_V1`]/[`alpn::HARMONY_TUNNEL_V2`]) and an in-protocol
//! *hello* frame (common, additive feature negotiation; [`TunnelHello`]).
//!
//! The client's `ProtocolCompatRegistry` (the loud per-peer incompatibility
//! surface) does NOT live here: it was the app-coupled Network Health sink, and
//! is replaced in this crate by the narrow [`crate::manager::CompatSink`] seam.
//! The pure hello encode/decode/gate helpers below are wire-format-load-bearing
//! and carried over byte-for-byte.

/// ALPN wire strings for the tunnel protocol generations. Carried from the
/// client's `iroh_endpoint::alpn` module — the exact byte-strings the endpoint
/// advertises and the initiator dials, preserved verbatim.
pub mod alpn {
    /// First tunnel ALPN generation.
    pub const HARMONY_TUNNEL_V1: &[u8] = b"harmony/tunnel/v1";
    /// Second (current) tunnel ALPN generation — adds the versioned hello frame.
    pub const HARMONY_TUNNEL_V2: &[u8] = b"harmony/tunnel/v2";
}

// The ALPN-generation policy constants (TUNNEL_ALPN_GENERATION /
// MIN_SUPPORTED_TUNNEL_ALPN_GENERATION) that governed the client's *endpoint
// bind list* are intentionally NOT carried here: this crate does not own the
// bind list (the app / endpoint substrate does). Only the wire strings above
// and the hello-version constants below are needed by the tunnel driver.

/// Version carried in the tunnel [`TunnelHello`] frame — the *feature* rate of
/// change, orthogonal to the ALPN generation.
pub const TUNNEL_PROTOCOL_VERSION: u16 = 1;

/// Oldest tunnel hello `protocol_version` this build interoperates with.
pub const MIN_SUPPORTED_TUNNEL_PROTOCOL_VERSION: u16 = 1;

/// Hard cap on an encoded [`TunnelHello`] frame. [`decode_hello`] refuses any
/// input above this before parsing, so a hostile peer can't force an unbounded
/// allocation on the first frame.
pub const TUNNEL_HELLO_MAX: usize = 1024;

/// First frame each side sends on a freshly opened `/v2` tunnel stream.
/// `capabilities` is an additive bitmap: unknown bits are ignored, so a newer
/// peer's extra features never break an older peer. New fields MUST be
/// `#[serde(default)]` — never `deny_unknown_fields`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TunnelHello {
    /// Feature version of the tunnel protocol the sender speaks.
    pub protocol_version: u16,
    /// Additive capability bitmap; unknown bits are ignored by the receiver.
    #[serde(default)]
    pub capabilities: u64,
}

impl TunnelHello {
    /// The hello this build advertises: [`TUNNEL_PROTOCOL_VERSION`] with no
    /// optional capabilities set yet.
    pub fn current() -> Self {
        Self {
            protocol_version: TUNNEL_PROTOCOL_VERSION,
            capabilities: 0,
        }
    }
}

/// Encode a [`TunnelHello`] to CBOR bytes (no length prefix; the caller frames
/// it). Preserves the client's `ciborium::into_writer` wire encoding.
pub fn encode_hello(h: &TunnelHello) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    ciborium::into_writer(h, &mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// Decode a [`TunnelHello`] from CBOR bytes, bounding the input at
/// [`TUNNEL_HELLO_MAX`] before parsing. Unknown fields are tolerated and a
/// missing `capabilities` defaults to `0`, so a v-next hello still decodes.
pub fn decode_hello(bytes: &[u8]) -> Result<TunnelHello, String> {
    if bytes.len() > TUNNEL_HELLO_MAX {
        return Err(format!(
            "tunnel hello {} bytes exceeds max {}",
            bytes.len(),
            TUNNEL_HELLO_MAX
        ));
    }
    ciborium::from_reader(bytes).map_err(|e| e.to_string())
}

/// Compatibility gate for a received hello. `Err(reason)` when the peer's
/// `protocol_version` is below [`MIN_SUPPORTED_TUNNEL_PROTOCOL_VERSION`]; a
/// version newer than ours is always compatible (unknown capability bits are
/// ignored). The returned reason is what the caller records via the
/// [`CompatSink`](crate::manager::CompatSink).
pub fn check_hello_compatible(h: &TunnelHello) -> Result<(), String> {
    if h.protocol_version < MIN_SUPPORTED_TUNNEL_PROTOCOL_VERSION {
        return Err(format!(
            "tunnel hello v{} < min supported v{}",
            h.protocol_version, MIN_SUPPORTED_TUNNEL_PROTOCOL_VERSION
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_roundtrips_via_cbor() {
        let h = TunnelHello {
            protocol_version: 1,
            capabilities: 0b101,
        };
        let bytes = encode_hello(&h).unwrap();
        assert!(bytes.len() < TUNNEL_HELLO_MAX);
        assert_eq!(decode_hello(&bytes).unwrap(), h);
    }

    #[test]
    fn hello_decode_tolerates_unknown_fields_and_missing_capabilities() {
        let mut extended = std::collections::BTreeMap::new();
        extended.insert(
            "protocol_version".to_string(),
            ciborium::Value::Integer(7.into()),
        );
        extended.insert(
            "some_future_field".to_string(),
            ciborium::Value::Text("x".into()),
        );
        let mut bytes = Vec::new();
        ciborium::into_writer(
            &ciborium::Value::Map(
                extended
                    .into_iter()
                    .map(|(k, v)| (ciborium::Value::Text(k), v))
                    .collect(),
            ),
            &mut bytes,
        )
        .unwrap();
        let h = decode_hello(&bytes).unwrap();
        assert_eq!(h.protocol_version, 7);
        assert_eq!(h.capabilities, 0);
    }

    #[test]
    fn decode_hello_rejects_oversized_frame() {
        assert!(decode_hello(&vec![0u8; TUNNEL_HELLO_MAX + 1]).is_err());
    }

    #[test]
    fn check_hello_rejects_below_min_supported() {
        assert!(check_hello_compatible(&TunnelHello {
            protocol_version: 0,
            capabilities: 0
        })
        .is_err());
        assert!(check_hello_compatible(&TunnelHello::current()).is_ok());
        // A NEWER version than ours is compatible (unknown capability bits ignored).
        assert!(check_hello_compatible(&TunnelHello {
            protocol_version: u16::MAX,
            capabilities: u64::MAX
        })
        .is_ok());
    }

    #[test]
    fn tunnel_alpn_generations_are_the_shipped_wire_strings() {
        assert_eq!(alpn::HARMONY_TUNNEL_V1, b"harmony/tunnel/v1");
        assert_eq!(alpn::HARMONY_TUNNEL_V2, b"harmony/tunnel/v2");
    }
}

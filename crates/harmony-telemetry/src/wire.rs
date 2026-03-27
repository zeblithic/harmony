//! Wire format encode/decode with tag-byte prefix.

use crate::types::TelemetryEvent;

/// Tag byte for JSON-encoded payloads.
pub const JSON_TAG: u8 = 0x00;

/// Errors from wire decode operations.
#[derive(Debug)]
pub enum TelemetryError {
    /// Payload was empty (no tag byte).
    EmptyPayload,
    /// Unknown wire format tag.
    UnsupportedFormat(u8),
    /// JSON deserialization failed.
    Json(serde_json::Error),
}

impl core::fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyPayload => write!(f, "empty payload"),
            Self::UnsupportedFormat(tag) => write!(f, "unsupported format tag: 0x{tag:02x}"),
            Self::Json(e) => write!(f, "json error: {e}"),
        }
    }
}

impl std::error::Error for TelemetryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}

fn check_tag(payload: &[u8]) -> Result<(), TelemetryError> {
    if payload.is_empty() {
        return Err(TelemetryError::EmptyPayload);
    }
    if payload[0] != JSON_TAG {
        return Err(TelemetryError::UnsupportedFormat(payload[0]));
    }
    Ok(())
}

/// Encode a TelemetryEvent to wire format: [JSON_TAG][json_bytes].
pub fn encode_event(event: &TelemetryEvent) -> Result<Vec<u8>, serde_json::Error> {
    let json = serde_json::to_vec(event)?;
    let mut payload = Vec::with_capacity(1 + json.len());
    payload.push(JSON_TAG);
    payload.extend_from_slice(&json);
    Ok(payload)
}

/// Decode a TelemetryEvent from wire format.
pub fn decode_event(payload: &[u8]) -> Result<TelemetryEvent, TelemetryError> {
    check_tag(payload)?;
    serde_json::from_slice(&payload[1..]).map_err(TelemetryError::Json)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event() -> TelemetryEvent {
        TelemetryEvent {
            node_addr: "deadbeef".to_string(),
            intent: "anomaly".to_string(),
            sequence: 7,
            timestamp: 1711500000,
            payload: serde_json::json!({"severity": "high"}),
            confidence: Some(0.88),
            source: Some("qwen3vl-4b".to_string()),
        }
    }

    #[test]
    fn event_wire_round_trip() {
        let event = sample_event();
        let encoded = encode_event(&event).unwrap();
        assert_eq!(encoded[0], JSON_TAG);
        let decoded = decode_event(&encoded).unwrap();
        assert_eq!(decoded.node_addr, "deadbeef");
        assert_eq!(decoded.intent, "anomaly");
        assert_eq!(decoded.sequence, 7);
        assert_eq!(decoded.timestamp, 1711500000);
        assert_eq!(decoded.payload, serde_json::json!({"severity": "high"}));
        assert_eq!(decoded.confidence, Some(0.88));
        assert_eq!(decoded.source.as_deref(), Some("qwen3vl-4b"));
    }

    #[test]
    fn decode_empty_payload() {
        assert!(matches!(decode_event(&[]), Err(TelemetryError::EmptyPayload)));
    }

    #[test]
    fn decode_unknown_tag() {
        assert!(matches!(
            decode_event(&[0xFF, b'{', b'}']),
            Err(TelemetryError::UnsupportedFormat(0xFF))
        ));
    }

    #[test]
    fn decode_invalid_json() {
        assert!(matches!(
            decode_event(&[JSON_TAG, b'n', b'o', b't']),
            Err(TelemetryError::Json(_))
        ));
    }

    #[test]
    fn error_display_variants() {
        let e1 = TelemetryError::EmptyPayload;
        assert_eq!(e1.to_string(), "empty payload");

        let e2 = TelemetryError::UnsupportedFormat(0xAB);
        assert_eq!(e2.to_string(), "unsupported format tag: 0xab");

        let bad = decode_event(&[JSON_TAG, b'{']).unwrap_err();
        let msg = bad.to_string();
        assert!(msg.starts_with("json error:"), "got: {msg}");
    }
}

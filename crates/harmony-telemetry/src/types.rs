//! Telemetry event types.

use serde::{Deserialize, Serialize};

/// A structured telemetry event published by a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    /// Node address (lowercase hex) that produced this event.
    pub node_addr: String,
    /// Intent tag — what kind of observation (e.g. "object_detected", "health").
    pub intent: String,
    /// Monotonic event sequence number (per-node, 0-indexed).
    pub sequence: u64,
    /// Unix epoch seconds when the event was produced.
    pub timestamp: u64,
    /// Intent-specific payload.
    pub payload: serde_json::Value,
    /// Optional confidence score (0.0–1.0) for ML-derived events.
    pub confidence: Option<f32>,
    /// Optional source descriptor (e.g. "camera:0", "cpu_monitor", "qwen3vl-4b").
    pub source: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn telemetry_event_round_trip() {
        let event = TelemetryEvent {
            node_addr: "deadbeef01020304".to_string(),
            intent: "object_detected".to_string(),
            sequence: 42,
            timestamp: 1711500000,
            payload: json!({"class": "person", "bbox": [10, 20, 100, 200]}),
            confidence: Some(0.95),
            source: Some("camera:0".to_string()),
        };
        let json = serde_json::to_string(&event).unwrap();
        let decoded: TelemetryEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.node_addr, "deadbeef01020304");
        assert_eq!(decoded.intent, "object_detected");
        assert_eq!(decoded.sequence, 42);
        assert_eq!(decoded.timestamp, 1711500000);
        assert_eq!(decoded.payload, json!({"class": "person", "bbox": [10, 20, 100, 200]}));
        assert_eq!(decoded.confidence, Some(0.95));
        assert_eq!(decoded.source.as_deref(), Some("camera:0"));
    }

    #[test]
    fn telemetry_event_optional_fields_absent() {
        let event = TelemetryEvent {
            node_addr: "aabbccdd".to_string(),
            intent: "health".to_string(),
            sequence: 0,
            timestamp: 1711500000,
            payload: json!({"cpu_percent": 45.2, "mem_mb": 1024}),
            confidence: None,
            source: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        let decoded: TelemetryEvent = serde_json::from_str(&json).unwrap();
        assert!(decoded.confidence.is_none());
        assert!(decoded.source.is_none());
        assert_eq!(decoded.intent, "health");
    }

    #[test]
    fn telemetry_event_varied_payloads() {
        let event = TelemetryEvent {
            node_addr: "aa".to_string(),
            intent: "ping".to_string(),
            sequence: 0,
            timestamp: 0,
            payload: serde_json::Value::Null,
            confidence: None,
            source: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        let decoded: TelemetryEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.payload, serde_json::Value::Null);

        let event2 = TelemetryEvent {
            node_addr: "bb".to_string(),
            intent: "complex".to_string(),
            sequence: 1,
            timestamp: 1,
            payload: json!({"nested": {"array": [1, 2, 3], "flag": true}}),
            confidence: None,
            source: None,
        };
        let json2 = serde_json::to_string(&event2).unwrap();
        let decoded2: TelemetryEvent = serde_json::from_str(&json2).unwrap();
        assert_eq!(decoded2.payload, json!({"nested": {"array": [1, 2, 3], "flag": true}}));
    }

    #[test]
    fn telemetry_event_snake_case_keys() {
        let event = TelemetryEvent {
            node_addr: "aa".to_string(),
            intent: "test".to_string(),
            sequence: 0,
            timestamp: 0,
            payload: json!({}),
            confidence: Some(0.5),
            source: Some("test".to_string()),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"node_addr\""), "expected snake_case, got: {json}");
        assert!(json.contains("\"intent\""), "expected intent key, got: {json}");
        assert!(!json.contains("\"nodeAddr\""), "unexpected camelCase in: {json}");
    }
}

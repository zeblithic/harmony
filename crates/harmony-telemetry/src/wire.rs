//! Wire format encode/decode (implemented in Task 2).

use crate::TelemetryEvent;

/// Errors that can occur during wire encode/decode.
#[derive(Debug)]
pub enum TelemetryError {
    /// JSON serialization/deserialization error.
    Json(serde_json::Error),
}

impl From<serde_json::Error> for TelemetryError {
    fn from(e: serde_json::Error) -> Self {
        TelemetryError::Json(e)
    }
}

/// Encode a `TelemetryEvent` to JSON bytes.
pub fn encode_event(event: &TelemetryEvent) -> Result<Vec<u8>, TelemetryError> {
    Ok(serde_json::to_vec(event)?)
}

/// Decode a `TelemetryEvent` from JSON bytes.
pub fn decode_event(bytes: &[u8]) -> Result<TelemetryEvent, TelemetryError> {
    Ok(serde_json::from_slice(bytes)?)
}

//! Wire format encode/decode with tag-byte prefix.

use crate::types::{AgentCapacity, AgentResult, AgentTask, StreamChunk};

/// Tag byte for JSON-encoded payloads.
pub const JSON_TAG: u8 = 0x00;

/// Errors from wire decode operations.
#[derive(Debug)]
pub enum AgentError {
    /// Payload was empty (no tag byte).
    EmptyPayload,
    /// Unknown wire format tag.
    UnsupportedFormat(u8),
    /// JSON deserialization failed.
    Json(serde_json::Error),
}

impl core::fmt::Display for AgentError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyPayload => write!(f, "empty payload"),
            Self::UnsupportedFormat(tag) => write!(f, "unsupported format tag: 0x{tag:02x}"),
            Self::Json(e) => write!(f, "json error: {e}"),
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}

fn check_tag(payload: &[u8]) -> Result<(), AgentError> {
    if payload.is_empty() {
        return Err(AgentError::EmptyPayload);
    }
    if payload[0] != JSON_TAG {
        return Err(AgentError::UnsupportedFormat(payload[0]));
    }
    Ok(())
}

/// Encode an AgentTask to wire format: [JSON_TAG][json_bytes].
pub fn encode_task(task: &AgentTask) -> Result<Vec<u8>, serde_json::Error> {
    let json = serde_json::to_vec(task)?;
    let mut payload = Vec::with_capacity(1 + json.len());
    payload.push(JSON_TAG);
    payload.extend_from_slice(&json);
    Ok(payload)
}

/// Decode an AgentTask from wire format.
pub fn decode_task(payload: &[u8]) -> Result<AgentTask, AgentError> {
    check_tag(payload)?;
    serde_json::from_slice(&payload[1..]).map_err(AgentError::Json)
}

/// Encode an AgentResult to wire format.
pub fn encode_result(result: &AgentResult) -> Result<Vec<u8>, serde_json::Error> {
    let json = serde_json::to_vec(result)?;
    let mut payload = Vec::with_capacity(1 + json.len());
    payload.push(JSON_TAG);
    payload.extend_from_slice(&json);
    Ok(payload)
}

/// Decode an AgentResult from wire format.
pub fn decode_result(payload: &[u8]) -> Result<AgentResult, AgentError> {
    check_tag(payload)?;
    serde_json::from_slice(&payload[1..]).map_err(AgentError::Json)
}

/// Encode an AgentCapacity to wire format.
pub fn encode_capacity(cap: &AgentCapacity) -> Result<Vec<u8>, serde_json::Error> {
    let json = serde_json::to_vec(cap)?;
    let mut payload = Vec::with_capacity(1 + json.len());
    payload.push(JSON_TAG);
    payload.extend_from_slice(&json);
    Ok(payload)
}

/// Decode an AgentCapacity from wire format.
pub fn decode_capacity(payload: &[u8]) -> Result<AgentCapacity, AgentError> {
    check_tag(payload)?;
    serde_json::from_slice(&payload[1..]).map_err(AgentError::Json)
}

/// Encode a StreamChunk to wire format: [JSON_TAG][json_bytes].
pub fn encode_chunk(chunk: &StreamChunk) -> Result<Vec<u8>, serde_json::Error> {
    let json = serde_json::to_vec(chunk)?;
    let mut payload = Vec::with_capacity(1 + json.len());
    payload.push(JSON_TAG);
    payload.extend_from_slice(&json);
    Ok(payload)
}

/// Decode a StreamChunk from wire format.
///
/// Returns `AgentError::Json` for deserialization failures, and
/// `AgentError::EmptyPayload` or `AgentError::UnsupportedFormat`
/// for wire-level issues. This asymmetry (encode returns `serde_json::Error`,
/// decode returns `AgentError`) matches the existing encode/decode pattern.
pub fn decode_chunk(payload: &[u8]) -> Result<StreamChunk, AgentError> {
    check_tag(payload)?;
    serde_json::from_slice(&payload[1..]).map_err(AgentError::Json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn task_wire_round_trip() {
        let task = AgentTask {
            task_id: "t1".to_string(),
            task_type: "inference".to_string(),
            params: serde_json::json!({"prompt": "Hello"}),
            context: None,
        };
        let encoded = encode_task(&task).unwrap();
        assert_eq!(encoded[0], JSON_TAG);
        let decoded = decode_task(&encoded).unwrap();
        assert_eq!(decoded.task_id, "t1");
    }

    #[test]
    fn result_wire_round_trip() {
        let result = AgentResult {
            task_id: "t1".to_string(),
            status: TaskStatus::Success,
            output: Some(serde_json::json!({"text": "output"})),
            error: None,
        };
        let encoded = encode_result(&result).unwrap();
        assert_eq!(encoded[0], JSON_TAG);
        let decoded = decode_result(&encoded).unwrap();
        assert_eq!(decoded.status, TaskStatus::Success);
    }

    #[test]
    fn capacity_wire_round_trip() {
        let cap = AgentCapacity {
            agent_id: "deadbeef".to_string(),
            task_types: vec!["inference".to_string()],
            status: AgentStatus::Ready,
            max_concurrent: 2,
        };
        let encoded = encode_capacity(&cap).unwrap();
        assert_eq!(encoded[0], JSON_TAG);
        let decoded = decode_capacity(&encoded).unwrap();
        assert_eq!(decoded.agent_id, "deadbeef");
    }

    #[test]
    fn decode_empty_payload_returns_error() {
        assert!(matches!(decode_task(&[]), Err(AgentError::EmptyPayload)));
    }

    #[test]
    fn decode_unknown_tag_returns_error() {
        assert!(matches!(decode_task(&[0xFF, b'{', b'}']), Err(AgentError::UnsupportedFormat(0xFF))));
    }

    #[test]
    fn decode_invalid_json_returns_error() {
        assert!(matches!(decode_task(&[JSON_TAG, b'n', b'o', b't']), Err(AgentError::Json(_))));
    }

    #[test]
    fn chunk_wire_round_trip() {
        let chunk = StreamChunk {
            task_id: "task-wire".to_string(),
            sequence: 7,
            payload: serde_json::json!({"token": "world"}),
            final_chunk: false,
        };
        let encoded = encode_chunk(&chunk).unwrap();
        assert_eq!(encoded[0], JSON_TAG);
        let decoded = decode_chunk(&encoded).unwrap();
        assert_eq!(decoded.task_id, "task-wire");
        assert_eq!(decoded.sequence, 7);
        assert_eq!(decoded.payload, serde_json::json!({"token": "world"}));
        assert!(!decoded.final_chunk);
    }

    #[test]
    fn chunk_wire_final_round_trip() {
        let chunk = StreamChunk {
            task_id: "task-final".to_string(),
            sequence: 99,
            payload: serde_json::json!({"summary": "done"}),
            final_chunk: true,
        };
        let encoded = encode_chunk(&chunk).unwrap();
        let decoded = decode_chunk(&encoded).unwrap();
        assert_eq!(decoded.task_id, "task-final");
        assert_eq!(decoded.sequence, 99);
        assert_eq!(decoded.payload, serde_json::json!({"summary": "done"}));
        assert!(decoded.final_chunk);
    }

    #[test]
    fn chunk_decode_empty_payload() {
        assert!(matches!(decode_chunk(&[]), Err(AgentError::EmptyPayload)));
    }

    #[test]
    fn chunk_decode_unknown_tag() {
        assert!(matches!(
            decode_chunk(&[0xFF, b'{', b'}']),
            Err(AgentError::UnsupportedFormat(0xFF))
        ));
    }

    #[test]
    fn chunk_decode_invalid_json() {
        assert!(matches!(
            decode_chunk(&[JSON_TAG, b'n', b'o', b't']),
            Err(AgentError::Json(_))
        ));
    }
}

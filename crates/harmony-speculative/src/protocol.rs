//! Verify request/response payload serialization.
//!
//! Wire formats:
//!
//! **VerifyRequest:**
//! ```text
//! [0x04] [context_len: u32 LE] [context_tokens: u32[] LE] [draft_count: u8] [draft_entries...]
//! ```
//! Each draft entry is `[token_id: u32 LE] [logprob: f32 LE]` (8 bytes).
//!
//! **VerifyResponse (success):**
//! ```text
//! [0x00] [accepted_count: u8] [bonus_token: u32 LE] [bonus_logprob: f32 LE]
//! ```
//!
//! **VerifyResponse (error):**
//! ```text
//! [0x01] [error_message_utf8]
//! ```

use crate::{DraftEntry, VerifyRequest, VerifyResponse, VERIFY_TAG};

/// Size in bytes of a serialized draft entry (token_id u32 + logprob f32).
const DRAFT_ENTRY_SIZE: usize = 8;

/// Tag byte for a successful verify response.
const RESPONSE_OK_TAG: u8 = 0x00;

/// Tag byte for an error verify response.
const RESPONSE_ERR_TAG: u8 = 0x01;

impl VerifyRequest {
    /// Serialize this request into a byte vector.
    pub fn serialize(&self) -> Vec<u8> {
        assert!(
            self.drafts.len() <= u8::MAX as usize,
            "draft count {} exceeds u8::MAX",
            self.drafts.len()
        );
        assert!(
            self.context_tokens.len() <= u32::MAX as usize,
            "context length {} exceeds u32::MAX",
            self.context_tokens.len()
        );
        let context_len = self.context_tokens.len() as u32;
        let draft_count = self.drafts.len() as u8;

        // 1 (tag) + 4 (context_len) + 4*context_len + 1 (draft_count) + 8*draft_count
        let capacity =
            1 + 4 + (self.context_tokens.len() * 4) + 1 + (self.drafts.len() * DRAFT_ENTRY_SIZE);
        let mut buf = Vec::with_capacity(capacity);

        buf.push(VERIFY_TAG);
        buf.extend_from_slice(&context_len.to_le_bytes());
        for &tok in &self.context_tokens {
            buf.extend_from_slice(&tok.to_le_bytes());
        }
        buf.push(draft_count);
        for draft in &self.drafts {
            buf.extend_from_slice(&draft.token_id.to_le_bytes());
            buf.extend_from_slice(&draft.logprob.to_le_bytes());
        }

        buf
    }

    /// Parse a verify request from a byte payload.
    pub fn parse(payload: &[u8]) -> Result<Self, String> {
        if payload.is_empty() {
            return Err("empty payload".into());
        }
        if payload[0] != VERIFY_TAG {
            return Err(format!(
                "wrong tag: expected 0x{VERIFY_TAG:02x}, got 0x{:02x}",
                payload[0]
            ));
        }

        let mut pos = 1;

        // context_len
        if payload.len() < pos + 4 {
            return Err("truncated payload: missing context_len".into());
        }
        let context_len =
            u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        // Guard against overflow on 32-bit targets: context_len * 4 must not wrap.
        const MAX_CONTEXT_TOKENS: usize = 32_768;
        if context_len > MAX_CONTEXT_TOKENS {
            return Err(format!(
                "context_len {context_len} exceeds maximum {MAX_CONTEXT_TOKENS}"
            ));
        }

        // context_tokens
        let context_bytes = context_len * 4;
        if payload.len() < pos + context_bytes {
            return Err("truncated payload: missing context tokens".into());
        }
        let mut context_tokens = Vec::with_capacity(context_len);
        for _ in 0..context_len {
            let tok = u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap());
            context_tokens.push(tok);
            pos += 4;
        }

        // draft_count
        if payload.len() < pos + 1 {
            return Err("truncated payload: missing draft_count".into());
        }
        let draft_count = payload[pos] as usize;
        pos += 1;

        // draft entries
        let drafts_bytes = draft_count * DRAFT_ENTRY_SIZE;
        if payload.len() < pos + drafts_bytes {
            return Err("truncated payload: missing draft entries".into());
        }
        let mut drafts = Vec::with_capacity(draft_count);
        for _ in 0..draft_count {
            let token_id = u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let logprob = f32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap());
            pos += 4;
            drafts.push(DraftEntry { token_id, logprob });
        }

        Ok(VerifyRequest {
            context_tokens,
            drafts,
        })
    }
}

impl VerifyResponse {
    /// Serialize a successful verify response into a byte vector.
    pub fn serialize(&self) -> Vec<u8> {
        // 1 (tag) + 1 (accepted_count) + 4 (bonus_token) + 4 (bonus_logprob) = 10
        let mut buf = Vec::with_capacity(10);
        buf.push(RESPONSE_OK_TAG);
        buf.push(self.accepted_count);
        buf.extend_from_slice(&self.bonus_token.to_le_bytes());
        buf.extend_from_slice(&self.bonus_logprob.to_le_bytes());
        buf
    }

    /// Serialize an error verify response from a message string.
    pub fn serialize_error(message: &str) -> Vec<u8> {
        let mut buf = Vec::with_capacity(1 + message.len());
        buf.push(RESPONSE_ERR_TAG);
        buf.extend_from_slice(message.as_bytes());
        buf
    }

    /// Parse a verify response from a byte payload.
    ///
    /// Returns `Err(message)` if the payload represents an error response
    /// (tag `0x01`) or if the payload is malformed.
    pub fn parse(payload: &[u8]) -> Result<Self, String> {
        if payload.is_empty() {
            return Err("empty payload".into());
        }

        match payload[0] {
            RESPONSE_OK_TAG => {
                // 1 (tag) + 1 (accepted_count) + 4 (bonus_token) + 4 (bonus_logprob) = 10
                if payload.len() < 10 {
                    return Err("truncated success response".into());
                }
                let accepted_count = payload[1];
                let bonus_token =
                    u32::from_le_bytes(payload[2..6].try_into().unwrap());
                let bonus_logprob =
                    f32::from_le_bytes(payload[6..10].try_into().unwrap());
                Ok(VerifyResponse {
                    accepted_count,
                    bonus_token,
                    bonus_logprob,
                })
            }
            RESPONSE_ERR_TAG => {
                let message = core::str::from_utf8(&payload[1..])
                    .map_err(|e| format!("invalid UTF-8 in error message: {e}"))?;
                Err(message.to_string())
            }
            tag => Err(format!("unknown response tag: 0x{tag:02x}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> VerifyRequest {
        VerifyRequest {
            context_tokens: vec![100, 200, 300],
            drafts: vec![
                DraftEntry {
                    token_id: 42,
                    logprob: -0.5,
                },
                DraftEntry {
                    token_id: 99,
                    logprob: -1.2,
                },
            ],
        }
    }

    fn sample_response() -> VerifyResponse {
        VerifyResponse {
            accepted_count: 2,
            bonus_token: 500,
            bonus_logprob: -0.3,
        }
    }

    #[test]
    fn request_roundtrip() {
        let req = sample_request();
        let bytes = req.serialize();
        let parsed = VerifyRequest::parse(&bytes).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn response_roundtrip() {
        let resp = sample_response();
        let bytes = resp.serialize();
        let parsed = VerifyResponse::parse(&bytes).unwrap();
        assert_eq!(resp, parsed);
    }

    #[test]
    fn error_response_roundtrip() {
        let msg = "model overloaded";
        let bytes = VerifyResponse::serialize_error(msg);
        let err = VerifyResponse::parse(&bytes).unwrap_err();
        assert_eq!(err, msg);
    }

    #[test]
    fn request_parse_wrong_tag() {
        let mut bytes = sample_request().serialize();
        bytes[0] = 0xFF;
        let err = VerifyRequest::parse(&bytes).unwrap_err();
        assert!(err.contains("wrong tag"));
    }

    #[test]
    fn response_parse_unknown_tag() {
        let bytes = vec![0xAB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let err = VerifyResponse::parse(&bytes).unwrap_err();
        assert!(err.contains("unknown response tag"));
    }

    #[test]
    fn request_parse_truncated() {
        // Only tag + partial context_len
        let bytes = vec![VERIFY_TAG, 0x01, 0x00];
        let err = VerifyRequest::parse(&bytes).unwrap_err();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn response_parse_truncated() {
        // Success tag but too short
        let bytes = vec![RESPONSE_OK_TAG, 0x02, 0x00];
        let err = VerifyResponse::parse(&bytes).unwrap_err();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn request_parse_empty() {
        let err = VerifyRequest::parse(&[]).unwrap_err();
        assert!(err.contains("empty"));
    }

    #[test]
    fn response_parse_empty() {
        let err = VerifyResponse::parse(&[]).unwrap_err();
        assert!(err.contains("empty"));
    }

    #[test]
    fn request_zero_context() {
        let req = VerifyRequest {
            context_tokens: vec![],
            drafts: vec![DraftEntry {
                token_id: 7,
                logprob: -0.1,
            }],
        };
        let bytes = req.serialize();
        let parsed = VerifyRequest::parse(&bytes).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn request_zero_drafts() {
        let req = VerifyRequest {
            context_tokens: vec![1, 2, 3],
            drafts: vec![],
        };
        let bytes = req.serialize();
        let parsed = VerifyRequest::parse(&bytes).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn request_large_context_roundtrip() {
        let req = VerifyRequest {
            context_tokens: (0..1000).collect(),
            drafts: vec![
                DraftEntry {
                    token_id: 1,
                    logprob: -0.01,
                },
                DraftEntry {
                    token_id: 2,
                    logprob: -0.02,
                },
                DraftEntry {
                    token_id: 3,
                    logprob: -0.03,
                },
            ],
        };
        let bytes = req.serialize();
        let parsed = VerifyRequest::parse(&bytes).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn request_truncated_context_tokens() {
        // Says 2 context tokens but only has bytes for 1
        let mut bytes = vec![VERIFY_TAG];
        bytes.extend_from_slice(&2u32.to_le_bytes()); // context_len = 2
        bytes.extend_from_slice(&42u32.to_le_bytes()); // only 1 token
        let err = VerifyRequest::parse(&bytes).unwrap_err();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn request_truncated_draft_entries() {
        // Valid context but truncated draft entries
        let mut bytes = vec![VERIFY_TAG];
        bytes.extend_from_slice(&0u32.to_le_bytes()); // context_len = 0
        bytes.push(2); // draft_count = 2
        // Only 4 bytes (half a draft entry)
        bytes.extend_from_slice(&42u32.to_le_bytes());
        let err = VerifyRequest::parse(&bytes).unwrap_err();
        assert!(err.contains("truncated"));
    }

    /// Roundtrip with realistic sizes, asserting exact wire byte counts.
    ///
    /// Wire layout:
    ///   Request: 1 (tag) + 4 (context_len) + 4*1000 (tokens) + 1 (draft_count) + 8*5 (drafts)
    ///          = 1 + 4 + 4000 + 1 + 40 = 4046 bytes
    ///   Response: 1 (tag) + 1 (accepted_count) + 4 (bonus_token) + 4 (bonus_logprob) = 10 bytes
    #[test]
    fn protocol_roundtrip_realistic() {
        let request = VerifyRequest {
            context_tokens: (0..1000).collect(),
            drafts: (0..5)
                .map(|i| DraftEntry {
                    token_id: 100 + i,
                    logprob: -(i as f32 + 1.0) * 0.1,
                })
                .collect(),
        };
        let bytes = request.serialize();
        // 1 + 4 + 4000 + 1 + 40 = 4046
        assert_eq!(bytes.len(), 4046, "serialized request size mismatch");
        let parsed = VerifyRequest::parse(&bytes).unwrap();
        assert_eq!(request, parsed);

        let response = VerifyResponse {
            accepted_count: 3,
            bonus_token: 42,
            bonus_logprob: -0.5,
        };
        let resp_bytes = response.serialize();
        // 1 (ok_tag) + 1 (accepted_count) + 4 (bonus_token) + 4 (bonus_logprob) = 10
        assert_eq!(resp_bytes.len(), 10, "serialized response size mismatch");
        let parsed_resp = VerifyResponse::parse(&resp_bytes).unwrap();
        assert_eq!(response, parsed_resp);
    }
}

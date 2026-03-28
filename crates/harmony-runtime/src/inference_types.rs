// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Inference queryable types and payload parsing.
//!
//! This module contains the wire-format types shared between the runtime
//! orchestration layer and the harmony-node inference handlers.

/// Default maximum tokens to generate per inference request.
pub const DEFAULT_MAX_INFERENCE_TOKENS: u32 = 512;

/// Request payload tag for inference queries.
pub const INFERENCE_TAG: u8 = 0x02;

/// Request payload tag for token-level inference queries.
pub const TOKEN_INFERENCE_TAG: u8 = 0x03;

/// Maximum token count to prevent allocation bombs from untrusted input.
const MAX_INPUT_TOKENS: u32 = 131_072;

/// Capacity status bytes.
pub const CAPACITY_READY: u8 = 0x01;
pub const CAPACITY_BUSY: u8 = 0x00;

/// Parsed inference request from a Zenoh query payload.
pub struct InferenceRequest {
    /// The text prompt to generate from.
    pub prompt: String,
    /// 20-byte sampling parameters (temperature, top_p, top_k, repeat_penalty, repeat_last_n).
    pub sampling_params: [u8; 20],
}

impl InferenceRequest {
    /// Parse an inference request from a query payload tagged with 0x02.
    ///
    /// Format: `[0x02] [prompt_len: u32 LE] [prompt_utf8] [sampling_params: 20 bytes (optional)]`
    ///
    /// If sampling params are absent, greedy defaults are used (temperature=0.0).
    pub fn parse(payload: &[u8]) -> Result<Self, String> {
        if payload.is_empty() || payload[0] != INFERENCE_TAG {
            return Err(format!(
                "expected inference tag 0x{:02x}, got 0x{:02x}",
                INFERENCE_TAG,
                payload.first().copied().unwrap_or(0)
            ));
        }
        if payload.len() < 5 {
            return Err("payload too short for prompt length".into());
        }
        let prompt_len =
            u32::from_le_bytes([payload[1], payload[2], payload[3], payload[4]]) as usize;
        if payload.len() < 5 + prompt_len {
            return Err(format!(
                "payload too short: need {} bytes for prompt, have {}",
                prompt_len,
                payload.len() - 5
            ));
        }
        let prompt = std::str::from_utf8(&payload[5..5 + prompt_len])
            .map_err(|e| format!("invalid UTF-8 prompt: {e}"))?
            .to_string();

        let sampling_params = if payload.len() >= 5 + prompt_len + 20 {
            let mut params = [0u8; 20];
            params.copy_from_slice(&payload[5 + prompt_len..5 + prompt_len + 20]);
            params
        } else {
            greedy_defaults()
        };

        Ok(InferenceRequest {
            prompt,
            sampling_params,
        })
    }
}

/// Greedy sampling defaults: temperature=0.0, top_p=1.0, top_k=0, repeat_penalty=1.0, repeat_last_n=64.
fn greedy_defaults() -> [u8; 20] {
    let mut params = [0u8; 20];
    params[4..8].copy_from_slice(&1.0f32.to_le_bytes());
    params[12..16].copy_from_slice(&1.0f32.to_le_bytes());
    params[16..20].copy_from_slice(&64u32.to_le_bytes());
    params
}

/// Parsed token-level inference request from a Zenoh query payload.
pub struct TokenInferenceRequest {
    pub token_ids: Vec<u32>,
    pub sampling_params: [u8; 20],
}

impl TokenInferenceRequest {
    /// Parse a token-level inference request tagged with 0x03.
    ///
    /// Format: `[0x03] [token_count: u32 LE] [token_ids: u32 LE × count] [sampling_params: 20 bytes (optional)]`
    pub fn parse(payload: &[u8]) -> Result<Self, String> {
        if payload.is_empty() || payload[0] != TOKEN_INFERENCE_TAG {
            return Err(format!(
                "expected token inference tag 0x{:02x}, got 0x{:02x}",
                TOKEN_INFERENCE_TAG,
                payload.first().copied().unwrap_or(0)
            ));
        }
        if payload.len() < 5 {
            return Err("payload too short for token count".into());
        }
        let token_count = u32::from_le_bytes([payload[1], payload[2], payload[3], payload[4]]);
        if token_count == 0 {
            return Err("token count must be at least 1".into());
        }
        if token_count > MAX_INPUT_TOKENS {
            return Err(format!(
                "token count {} exceeds maximum {}",
                token_count, MAX_INPUT_TOKENS
            ));
        }
        let token_count = token_count as usize;
        let tokens_end = 5 + token_count * 4;
        if payload.len() < tokens_end {
            return Err(format!(
                "payload too short: need {} bytes for {} tokens, have {}",
                token_count * 4,
                token_count,
                payload.len() - 5
            ));
        }
        let mut token_ids = Vec::with_capacity(token_count);
        for i in 0..token_count {
            let offset = 5 + i * 4;
            token_ids.push(u32::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
            ]));
        }
        let sampling_params = if payload.len() >= tokens_end + 20 {
            let mut params = [0u8; 20];
            params.copy_from_slice(&payload[tokens_end..tokens_end + 20]);
            params
        } else {
            greedy_defaults()
        };
        Ok(TokenInferenceRequest {
            token_ids,
            sampling_params,
        })
    }
}

/// Input to the inference loop — either text (0x02) or pre-tokenized (0x03).
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceInput {
    /// Text prompt — will be tokenized by the engine.
    Text(String),
    /// Pre-tokenized token IDs — skip tokenization.
    TokenIds(Vec<u32>),
}

/// Build capacity advertisement payload.
///
/// Layout: `[model_gguf_cid:32] [status:u8]`
pub fn build_capacity_payload(model_cid: &[u8; 32], ready: bool) -> Vec<u8> {
    let mut payload = Vec::with_capacity(33);
    payload.extend_from_slice(model_cid);
    payload.push(if ready { CAPACITY_READY } else { CAPACITY_BUSY });
    payload
}

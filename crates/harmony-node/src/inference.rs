//! Inference queryable types and payload parsing.

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

/// Built-in inference runner WASM module (compiled from WAT by build.rs).
pub const INFERENCE_RUNNER_WASM: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/inference_runner.wasm"));

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

/// Output from the inference loop — matches the input type.
#[derive(Debug, Clone)]
pub enum InferenceOutput {
    /// Concatenated text from detokenized tokens.
    Text(String),
    /// Raw generated token IDs.
    TokenIds(Vec<u32>),
}

/// Convert raw 20-byte sampling parameters to `SamplingParams`.
///
/// Layout: `[temperature:f32 LE] [top_p:f32 LE] [top_k:u32 LE] [repeat_penalty:f32 LE] [repeat_last_n:u32 LE]`
#[cfg(feature = "inference")]
pub fn decode_sampling_params(raw: &[u8; 20]) -> harmony_inference::SamplingParams {
    let temperature = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let top_p = f32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let top_k = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
    let repeat_penalty = f32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]);
    let repeat_last_n = u32::from_le_bytes([raw[16], raw[17], raw[18], raw[19]]) as usize;
    harmony_inference::SamplingParams {
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        repeat_last_n,
    }
}

/// Build the WASM input for the inference runner module.
///
/// Layout: `[gguf_cid:32] [tokenizer_cid:32] [prompt_len:u32 LE] [prompt_utf8] [sampling_params:20] [max_tokens:u32 LE] [nonce:u64 LE]`
///
/// The trailing nonce ensures each request gets a unique `WorkflowId`,
/// preventing the WorkflowEngine from deduplicating non-deterministic
/// inference results (e.g. temperature > 0). The WASM module ignores it.
pub fn build_runner_input(
    gguf_cid: &[u8; 32],
    tokenizer_cid: &[u8; 32],
    request: &InferenceRequest,
    nonce: u64,
) -> Vec<u8> {
    let prompt_bytes = request.prompt.as_bytes();
    let prompt_len = prompt_bytes.len() as u32;
    let mut input = Vec::with_capacity(32 + 32 + 4 + prompt_bytes.len() + 20 + 4 + 8);
    input.extend_from_slice(gguf_cid);
    input.extend_from_slice(tokenizer_cid);
    input.extend_from_slice(&prompt_len.to_le_bytes());
    input.extend_from_slice(prompt_bytes);
    input.extend_from_slice(&request.sampling_params);
    input.extend_from_slice(&DEFAULT_MAX_INFERENCE_TOKENS.to_le_bytes());
    input.extend_from_slice(&nonce.to_le_bytes());
    input
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_inference_request_with_params() {
        let prompt = b"Hello world";
        let mut payload = vec![INFERENCE_TAG];
        payload.extend_from_slice(&(prompt.len() as u32).to_le_bytes());
        payload.extend_from_slice(prompt);
        let mut params = [0u8; 20];
        params[0..4].copy_from_slice(&0.7f32.to_le_bytes());
        params[4..8].copy_from_slice(&1.0f32.to_le_bytes());
        params[12..16].copy_from_slice(&1.0f32.to_le_bytes());
        params[16..20].copy_from_slice(&64u32.to_le_bytes());
        payload.extend_from_slice(&params);

        let req = InferenceRequest::parse(&payload).unwrap();
        assert_eq!(req.prompt, "Hello world");
        assert_eq!(
            f32::from_le_bytes([
                req.sampling_params[0],
                req.sampling_params[1],
                req.sampling_params[2],
                req.sampling_params[3]
            ]),
            0.7
        );
    }

    #[test]
    fn parse_inference_request_greedy_defaults() {
        let prompt = b"Test";
        let mut payload = vec![INFERENCE_TAG];
        payload.extend_from_slice(&(prompt.len() as u32).to_le_bytes());
        payload.extend_from_slice(prompt);

        let req = InferenceRequest::parse(&payload).unwrap();
        assert_eq!(req.prompt, "Test");
        assert_eq!(
            f32::from_le_bytes([
                req.sampling_params[0],
                req.sampling_params[1],
                req.sampling_params[2],
                req.sampling_params[3]
            ]),
            0.0
        );
        assert_eq!(
            f32::from_le_bytes([
                req.sampling_params[4],
                req.sampling_params[5],
                req.sampling_params[6],
                req.sampling_params[7]
            ]),
            1.0
        );
    }

    #[test]
    fn parse_inference_request_wrong_tag() {
        let payload = vec![0x00, 0, 0, 0, 0];
        assert!(InferenceRequest::parse(&payload).is_err());
    }

    #[test]
    fn parse_inference_request_truncated() {
        let payload = vec![INFERENCE_TAG, 0xFF, 0, 0, 0];
        assert!(InferenceRequest::parse(&payload).is_err());
    }

    #[test]
    fn parse_inference_request_empty() {
        assert!(InferenceRequest::parse(&[]).is_err());
    }

    #[test]
    fn build_runner_input_layout() {
        let gguf_cid = [0xAA; 32];
        let tok_cid = [0xBB; 32];
        let req = InferenceRequest {
            prompt: "hi".to_string(),
            sampling_params: [0u8; 20],
        };
        let input = build_runner_input(&gguf_cid, &tok_cid, &req, 42);
        assert_eq!(&input[0..32], &[0xAA; 32]);
        assert_eq!(&input[32..64], &[0xBB; 32]);
        assert_eq!(
            u32::from_le_bytes([input[64], input[65], input[66], input[67]]),
            2
        );
        assert_eq!(&input[68..70], b"hi");
        assert_eq!(input.len(), 32 + 32 + 4 + 2 + 20 + 4 + 8); // +8 for nonce

        // Nonce is at the end.
        let nonce_offset = input.len() - 8;
        assert_eq!(
            u64::from_le_bytes(input[nonce_offset..nonce_offset + 8].try_into().unwrap()),
            42
        );
        let mt_offset = nonce_offset - 4;
        assert_eq!(
            u32::from_le_bytes([
                input[mt_offset],
                input[mt_offset + 1],
                input[mt_offset + 2],
                input[mt_offset + 3]
            ]),
            DEFAULT_MAX_INFERENCE_TOKENS
        );
    }

    #[test]
    fn capacity_payload_ready() {
        let cid = [0xCC; 32];
        let payload = build_capacity_payload(&cid, true);
        assert_eq!(payload.len(), 33);
        assert_eq!(&payload[0..32], &[0xCC; 32]);
        assert_eq!(payload[32], CAPACITY_READY);
    }

    #[test]
    fn capacity_payload_busy() {
        let cid = [0xDD; 32];
        let payload = build_capacity_payload(&cid, false);
        assert_eq!(payload[32], CAPACITY_BUSY);
    }

    #[test]
    #[cfg(feature = "inference")]
    fn decode_sampling_params_greedy() {
        let mut raw = [0u8; 20];
        raw[4..8].copy_from_slice(&1.0f32.to_le_bytes()); // top_p
        raw[12..16].copy_from_slice(&1.0f32.to_le_bytes()); // repeat_penalty
        raw[16..20].copy_from_slice(&64u32.to_le_bytes()); // repeat_last_n

        let params = super::decode_sampling_params(&raw);
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.repeat_penalty, 1.0);
        assert_eq!(params.repeat_last_n, 64);
    }

    #[test]
    #[cfg(feature = "inference")]
    fn decode_sampling_params_custom() {
        let mut raw = [0u8; 20];
        raw[0..4].copy_from_slice(&0.7f32.to_le_bytes());
        raw[4..8].copy_from_slice(&0.9f32.to_le_bytes());
        raw[8..12].copy_from_slice(&40u32.to_le_bytes());
        raw[12..16].copy_from_slice(&1.1f32.to_le_bytes());
        raw[16..20].copy_from_slice(&128u32.to_le_bytes());

        let params = super::decode_sampling_params(&raw);
        assert!((params.temperature - 0.7).abs() < 1e-6);
        assert!((params.top_p - 0.9).abs() < 1e-6);
        assert_eq!(params.top_k, 40);
        assert!((params.repeat_penalty - 1.1).abs() < 1e-6);
        assert_eq!(params.repeat_last_n, 128);
    }

    #[test]
    fn parse_token_request_with_params() {
        let tokens: Vec<u32> = vec![100, 200, 300];
        let mut payload = vec![TOKEN_INFERENCE_TAG];
        payload.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for &t in &tokens {
            payload.extend_from_slice(&t.to_le_bytes());
        }
        let mut params = [0u8; 20];
        params[0..4].copy_from_slice(&0.7f32.to_le_bytes());
        params[4..8].copy_from_slice(&1.0f32.to_le_bytes());
        params[12..16].copy_from_slice(&1.0f32.to_le_bytes());
        params[16..20].copy_from_slice(&64u32.to_le_bytes());
        payload.extend_from_slice(&params);

        let req = TokenInferenceRequest::parse(&payload).unwrap();
        assert_eq!(req.token_ids, vec![100, 200, 300]);
        assert_eq!(
            f32::from_le_bytes([
                req.sampling_params[0],
                req.sampling_params[1],
                req.sampling_params[2],
                req.sampling_params[3]
            ]),
            0.7
        );
    }

    #[test]
    fn parse_token_request_greedy_defaults() {
        let tokens: Vec<u32> = vec![42];
        let mut payload = vec![TOKEN_INFERENCE_TAG];
        payload.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for &t in &tokens {
            payload.extend_from_slice(&t.to_le_bytes());
        }
        let req = TokenInferenceRequest::parse(&payload).unwrap();
        assert_eq!(req.token_ids, vec![42]);
        assert_eq!(
            f32::from_le_bytes([
                req.sampling_params[0],
                req.sampling_params[1],
                req.sampling_params[2],
                req.sampling_params[3]
            ]),
            0.0
        );
    }

    #[test]
    fn parse_token_request_wrong_tag() {
        let payload = vec![0x02, 0, 0, 0, 0];
        assert!(TokenInferenceRequest::parse(&payload).is_err());
    }

    #[test]
    fn parse_token_request_truncated() {
        let mut payload = vec![TOKEN_INFERENCE_TAG];
        payload.extend_from_slice(&100u32.to_le_bytes());
        assert!(TokenInferenceRequest::parse(&payload).is_err());
    }

    #[test]
    fn parse_token_request_empty_tokens() {
        let mut payload = vec![TOKEN_INFERENCE_TAG];
        payload.extend_from_slice(&0u32.to_le_bytes());
        assert!(TokenInferenceRequest::parse(&payload).is_err());
    }

    #[test]
    fn parse_token_request_too_many_tokens() {
        let mut payload = vec![TOKEN_INFERENCE_TAG];
        payload.extend_from_slice(&(super::MAX_INPUT_TOKENS + 1).to_le_bytes());
        assert!(TokenInferenceRequest::parse(&payload).is_err());
    }

    #[test]
    fn inference_runner_wasm_is_valid() {
        assert!(
            !INFERENCE_RUNNER_WASM.is_empty(),
            "inference runner WASM should not be empty"
        );
        assert_eq!(
            &INFERENCE_RUNNER_WASM[0..4],
            b"\0asm",
            "should start with WASM magic bytes"
        );
    }
}

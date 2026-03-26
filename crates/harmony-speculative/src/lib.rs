//! Decentralized Speculative Decoding (DSD) protocol for Harmony.
//!
//! Enables edge nodes with small draft models to generate candidate tokens
//! locally, send them to powerful mesh nodes for verification, achieving
//! ~2-3x speedup when network latency is lower than per-token compute time.

pub mod protocol;
pub mod verify;

/// Default number of draft tokens per verification round.
pub const DEFAULT_DRAFT_GAMMA: u8 = 5;

/// Payload tag for verify requests.
pub const VERIFY_TAG: u8 = 0x04;

/// A single draft token paired with its log-probability from the draft model.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftEntry {
    pub token_id: u32,
    pub logprob: f32,
}

/// Verify request — sent from edge to target.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyRequest {
    pub context_tokens: Vec<u32>,
    pub drafts: Vec<DraftEntry>,
}

/// Verify response — sent from target to edge.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyResponse {
    pub accepted_count: u8,
    pub bonus_token: u32,
    pub bonus_logprob: f32,
}

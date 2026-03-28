/// Errors from inference operations.
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    /// No model loaded — call `load_gguf()` first.
    #[error("no model loaded — call load_gguf() first")]
    ModelNotLoaded,

    /// No tokenizer loaded — call `load_tokenizer()` first.
    #[error("no tokenizer loaded — call load_tokenizer() first")]
    TokenizerNotLoaded,

    /// GGUF file is invalid or unsupported.
    #[error("invalid GGUF: {0}")]
    InvalidGguf(String),

    /// Tokenizer JSON is invalid.
    #[error("tokenizer error: {0}")]
    TokenizerError(String),

    /// Forward pass failed (tensor operation error).
    #[error("forward pass failed: {0}")]
    ForwardFailed(String),

    /// Sampling failed (empty logits, invalid params).
    #[error("sampling failed: {0}")]
    SamplingFailed(String),

    /// Cache does not match the loaded model architecture.
    #[error("cache mismatch: expected {expected} layers, got {actual}")]
    CacheMismatch { expected: usize, actual: usize },
}

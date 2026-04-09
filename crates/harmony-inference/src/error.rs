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

    /// Engram resolution failed (shard data missing or tensor error).
    #[error("engram resolution failed: {0}")]
    EngramResolutionFailed(String),

    /// KV cache compression or decompression failed.
    #[cfg(feature = "kv-compress")]
    #[error("compression failed: {0}")]
    CompressionFailed(String),

    /// Forward pass attempted while cache is compressed — call decompress() first.
    #[cfg(feature = "kv-compress")]
    #[error("cache is compressed — call decompress() before forward()")]
    CacheCompressed,

    /// Serialization or deserialization of compressed cache failed.
    #[cfg(feature = "kv-compress")]
    #[error("serialization failed: {0}")]
    SerializationFailed(String),
}

#[cfg(test)]
#[cfg(feature = "kv-compress")]
mod tests {
    use super::*;

    #[test]
    fn compression_failed_displays_message() {
        let err = InferenceError::CompressionFailed("bad tensor".into());
        assert_eq!(err.to_string(), "compression failed: bad tensor");
    }

    #[test]
    fn cache_compressed_displays_message() {
        let err = InferenceError::CacheCompressed;
        assert!(err.to_string().contains("compressed"));
    }

    #[test]
    fn serialization_failed_displays_message() {
        let err = InferenceError::SerializationFailed("bad data".into());
        assert_eq!(err.to_string(), "serialization failed: bad data");
    }
}
